// 最近命中着色器：采样命中三角形的材质、法线和运动信息，并写回 payload。
// 光照本身仍在 raygen 的 tracePath 中完成。
// tableAddr/entityTableAddr 直接从 push constant 读取，避免每次命中都多绕一次 WorldPush BDA。
import world_common;

[[vk::push_constant]] WorldPushConstants pc;

[[vk::binding(2, 0)]] Sampler2D blockAlbedoAtlas;
// 独立的 bindless 数组：分别存放每次绘制的实体反照率槽位，以及紧凑的规范材质页集合。
[[vk::binding(0, 1)]] Sampler2D entityAlbedoTex[];
[[vk::binding(1, 1)]] Sampler2D materialSurface0Tex[];
[[vk::binding(2, 1)]] Sampler2D materialNormalAoTex[];
[[vk::binding(3, 1)]] Sampler2D materialSurface1Tex[];

void payloadSetPacked(inout Payload payload, uint material, float roughness, float metalness,
        float emission, float sss, float ior, float transmission, uint emissionSource) {
    payload.flags = material | (emissionSource << PAYLOAD_EMISSION_SOURCE_SHIFT);
    payload.roughMetal = packHalf2(float2(roughness, metalness));
    payload.emissionSss = packHalf2(float2(emission, sss));
    payload.iorTransmission = packHalf2(float2(ior, transmission));
}

uint materialEmissionSource(MaterialHeader header, float emission) {
    if (emission <= 0.0) return EMISSION_SOURCE_NONE;
    if ((header.features & MATERIAL_FEATURE_OVERRIDE_EMISSION) != 0u) return EMISSION_SOURCE_OVERRIDE;
    if ((header.features & MATERIAL_FEATURE_SPEC) != 0u) return EMISSION_SOURCE_LAB_PBR;
    if ((header.features & MATERIAL_FEATURE_HEURISTIC_EMISSION) != 0u) return EMISSION_SOURCE_HEURISTIC;
    return EMISSION_SOURCE_UNIFORM;
}

static const float RAY_CONE_MIN_WIDTH = 1.0e-5;
static const float RAY_CONE_MAX_LOD = 12.0;

float2 texSizePx(Sampler2D t) {
    uint w, h;
    t.GetDimensions(w, h);
    return float2(w, h);
}

float rayConeHitWidth(uint rayCone) {
    float2 cone = unpackHalf2(rayCone);
    return max(cone.x + cone.y * max(RayTCurrent(), 0.0), RAY_CONE_MIN_WIDTH);
}

float edgeTexelsPerWorld(float3 p0, float3 p1, float2 uv0, float2 uv1, float2 textureSizePx) {
    return length((uv1 - uv0) * textureSizePx) / length(p1 - p0);
}

float rayConeTextureLod(uint rayCone, float2 textureSizePx, float3 p0, float3 p1, float3 p2, float2 uv0, float2 uv1, float2 uv2) {
    float texelsPerWorld = max(edgeTexelsPerWorld(p0, p1, uv0, uv1, textureSizePx),
            max(edgeTexelsPerWorld(p1, p2, uv1, uv2, textureSizePx),
                edgeTexelsPerWorld(p2, p0, uv2, uv0, textureSizePx)));
    float footprint = rayConeHitWidth(rayCone) * texelsPerWorld;
    return clamp(log2(footprint), 0.0, RAY_CONE_MAX_LOD);
}

float rayConeUnitUvLod(uint rayCone, float2 textureSizePx) {
    float footprint = rayConeHitWidth(rayCone) * max(textureSizePx.x, textureSizePx.y);
    return clamp(log2(footprint), 0.0, RAY_CONE_MAX_LOD);
}
// `n` 必须已朝向观察者（`vdir`）。将结果限制在地平线之上，避免掠射角扰动使法线穿过表面
// （修复黑斑）。实例变换只有平移，因此对象边
// 长等于世界空间边长。
float3 perturbNormal(float3 n, float3 p0, float3 p1, float3 p2, float2 t0, float2 t1, float2 t2, float3 vdir, float4 ntex) {
    float2 g1 = t1 - t0;
    float2 g2 = t2 - t0;
    float det = g1.x * g2.y - g1.y * g2.x;
    float r = 1.0 / det;
    float3 traw = ((p1 - p0) * g2.y - (p2 - p0) * g1.y) * r;
    float3 T = normalize(traw - n * dot(n, traw));       // 相对于朝向观察者的法线执行 Gram-Schmidt
    float3 B = cross(n, T) * (det < 0.0 ? -1.0 : 1.0);   // 根据 UV 绕序确定手性
    float2 nxy = ntex.xy * 2.0 - 1.0;
    float nz = sqrt(max(0.0, 1.0 - dot(nxy, nxy)));      // 重建 Z
    float3 nm = normalize(nxy.x * T + nxy.y * B + nz * n);
    float NoV = dot(nm, vdir);
    if (NoV < 0.02) {
        nm = normalize(nm + vdir * (0.02 - NoV));        // 保持在地平线之上（不翻转）
    }
    return nm;
}

struct Surface {
    float3 normal;
    float3 f0;
    float roughness;
    float metalness;
    float emission;
    float sss;
};

// 规范材质页使用一个紧凑的页索引，该索引由三个专用数组共享。
float4 samplePageSurface0(uint page, float2 uv, float lod) {
    return materialSurface0Tex[NonUniformResourceIndex(page)].SampleLevel(uv, lod);
}
float4 samplePageSurface1(uint page, float2 uv, float lod) {
    return materialSurface1Tex[NonUniformResourceIndex(page)].SampleLevel(uv, lod);
}
float4 samplePageNormalAo(uint page, float2 uv, float lod) {
    return materialNormalAoTex[NonUniformResourceIndex(page)].SampleLevel(uv, lod);
}

// 地形/实体材质的唯一评估路径。LabPBR 的分类字节已经
// 在 CPU 上解码；下面的每次采样都是可连续过滤的物理通道，并带有语义正确的 mip。
Surface evaluateMaterial(MaterialHeader header, float2 atlasUv, float lod, float3 albedo,
        float3 geometricNormal, float3 p0, float3 p1, float3 p2,
        float2 uv0, float2 uv1, float2 uv2, float3 vdir,
        float fallbackRoughness, float fallbackMetalness, float fallbackEmission, bool allowSss) {
    Surface surface;
    surface.normal = geometricNormal;
    surface.roughness = fallbackRoughness;
    surface.metalness = fallbackMetalness;
    surface.f0 = lerp(float3(0.04, 0.04, 0.04), albedo, fallbackMetalness);
    surface.emission = fallbackEmission;
    surface.sss = 0.0;

    float2 localUv = clamp((atlasUv - header.albedoUv.xy) * header.albedoUv.zw,
            float2(0.0, 0.0), float2(1.0, 1.0));
    float2 pageUv = header.materialUv.xy + localUv * header.materialUv.zw;
    float maxMaterialLod = float((header.features >> MATERIAL_MAX_LOD_SHIFT) & MATERIAL_MAX_LOD_MASK);
    float materialLod = min(lod, maxMaterialLod);
    uint page = header.texturePage;
    if ((header.features & (MATERIAL_FEATURE_SPEC | MATERIAL_FEATURE_HEURISTIC_EMISSION
            | MATERIAL_FEATURE_OVERRIDE_EMISSION)) != 0u) {
        float4 surface0 = samplePageSurface0(page, pageUv, materialLod);
        float4 surface1 = (header.features & (MATERIAL_FEATURE_SPEC | MATERIAL_FEATURE_OVERRIDE_EMISSION)) != 0u
                ? samplePageSurface1(page, pageUv, materialLod) : float4(0.0, 0.0, 0.0, 0.0);
        if ((header.features & MATERIAL_FEATURE_SPEC) != 0u) {
            surface.roughness = surface0.r;
            surface.metalness = surface0.g;
            surface.emission = surface0.b;
            surface.sss = allowSss ? surface0.a : 0.0;
            surface.f0 = surface1.rgb;
        } else {
            // 是否启用由方块状态语义编译得到；图元的精确光照等级
            // 提供强度，而材质页只提供归一化的空间遮罩。
            surface.emission = surface0.b * fallbackEmission;
        }
        if ((header.features & MATERIAL_FEATURE_OVERRIDE_EMISSION) != 0u) {
            float strength = float((header.features >> MATERIAL_EMISSION_STRENGTH_SHIFT)
                    & MATERIAL_EMISSION_STRENGTH_MASK)
                    * (MATERIAL_MAX_OVERRIDE_EMISSION_STRENGTH / float(MATERIAL_EMISSION_STRENGTH_MASK));
            surface.emission = surface1.a * strength;
        }
    }
    if ((header.features & MATERIAL_FEATURE_NORMAL) != 0u) {
        float4 normalTex = samplePageNormalAo(page, pageUv, materialLod);
        surface.normal = perturbNormal(geometricNormal, p0, p1, p2, uv0, uv1, uv2,
                vdir, normalTex);
    }
    return surface;
}

// 方块破坏覆盖层：由不透明地形以及彩色玻璃/冰共享。它重建命中位置，
// 在与破坏列表相同的重定位空间中，将位置轻推入实心方块以获得方块整数坐标，然后将局部 [0,1]
// 位置投影到法线最一致的轴对上（思路同 vanilla 的 SheetedDecalTextureGenerator），再以乘法混合
// 加入匹配的破坏阶段裂纹纹理，复现 vanilla 的碎裂混合（DST_COLOR*SRC_COLOR，再乘二）。
float3 applyBreaking(float3 albedo, uint rayCone, float3 rayOrigin, float3 rayDir, float hitT, float3 n) {
    ConstPtr<WorldPush> pcRef = ConstPtr<WorldPush>(pc.worldPushAddr);
    uint breakCount = pcRef[0].breakCount;
    if (breakCount == 0u) {
        return albedo;
    }
    float3 hitPos = rayOrigin + rayDir * hitT;
    int3 blockPos = int3(floor(hitPos - n * 0.01));
    for (uint bi = 0u; bi < breakCount; ++bi) {
        int4 ps = pcRef[0].breaking[bi].posSlot;
        if (all(ps.xyz == blockPos)) {
            float3 local = hitPos - float3(blockPos);
            float3 an = abs(n);
            float2 decalUv = (an.x >= an.y && an.x >= an.z) ? local.zy
                           : (an.y >= an.z) ? local.xz
                           : local.xy;
            float decalLod = rayConeUnitUvLod(rayCone, texSizePx(entityAlbedoTex[NonUniformResourceIndex(ps.w)]));
            float3 crack = srgbToLinear(entityAlbedoTex[NonUniformResourceIndex(ps.w)].SampleLevel(decalUv, decalLod).rgb);
            return clamp(crack * albedo, 0.0, 1.0);
        }
    }
    return albedo;
}

















[shader("closesthit")]
void main(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr) {
    float2 attribs = attr.barycentrics;
    uint rayCone = payload.rayCone;

    // 粒子 billboard：采样实体图集并写入漫反射 payload；镂空裁剪由 rahit 处理。
    if ((InstanceID() & PARTICLE_BIT) != 0u) {
        uint idx = InstanceID() & IDX_MASK;
        EntityGeom g = ConstPtr<EntityGeom>(pc.entityTableAddr)[idx];
        uint pid = PrimitiveIndex() + g.triBase[GeometryIndex()];
        Prim pr = ConstPtr<Prim>(g.primAddr)[pid];
        ConstPtr<uint> ib = ConstPtr<uint>(g.idxAddr);
        ConstPtr<float2> uvb = ConstPtr<float2>(g.uvAddr);
        uint p0 = ib[3u * pid + 0u];
        uint p1 = ib[3u * pid + 1u];
        uint p2 = ib[3u * pid + 2u];
        float3 pbary = float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
        float2 puv = pbary.x * uvb[p0] + pbary.y * uvb[p1] + pbary.z * uvb[p2];
        int pslot = int(pr.tint.w + 0.5);
        float3x3 o2w = (float3x3)ObjectToWorld3x4();
        float3 pp0 = mul(o2w, HitTriangleVertexPosition(0));
        float3 pp1 = mul(o2w, HitTriangleVertexPosition(1));
        float3 pp2 = mul(o2w, HitTriangleVertexPosition(2));
        float particleLod = rayConeTextureLod(rayCone, texSizePx(entityAlbedoTex[NonUniformResourceIndex(pslot)]),
                pp0, pp1, pp2, uvb[p0], uvb[p1], uvb[p2]);
        float3 pn = normalize(pr.normal.xyz);
        if (dot(pn, WorldRayDirection()) > 0.0) {
            pn = -pn;
        }
        payload.albedo = srgbToLinear(entityAlbedoTex[NonUniformResourceIndex(pslot)].SampleLevel(puv, particleLod).rgb) * pr.tint.rgb;
        payload.normal = pn;
        payload.hitT = RayTCurrent();
        // 粒子运动矢量：按重心坐标插值捕获的顶点位移；没有位移缓冲时回退到 rigidDisp。
        if (g.dispAddr != 0) {
            ConstPtr<float4> pd = ConstPtr<float4>(g.dispAddr);
            payload.motionPrev = pbary.x * pd[p0].xyz + pbary.y * pd[p1].xyz + pbary.z * pd[p2].xyz;
        } else {
            payload.motionPrev = g.rigidDisp.xyz;
        }
        payload.f0 = float3(0.0, 0.0, 0.0);
        payloadSetPacked(payload, MATERIAL_PARTICLE, 1.0, 0.0, 0.0, 0.0,
                1.0, 0.0, EMISSION_SOURCE_NONE);
        return;
    }
    // 动态实体：InstanceID 低位索引实体几何表，并从实体 bindless 图集采样材质。
    if ((InstanceID() & ENTITY_BIT) != 0u) {
        uint eidx = InstanceID() & ~ENTITY_BIT;
        EntityGeom g = ConstPtr<EntityGeom>(pc.entityTableAddr)[eidx];
        uint pid = PrimitiveIndex() + g.triBase[GeometryIndex()];
        Prim pr = ConstPtr<Prim>(g.primAddr)[pid];
        // 复用 BLAS 的实体可能在实例变换中带旋转，因此把对象空间法线转到世界空间。
        float3x3 entityO2w = (float3x3)ObjectToWorld3x4();
        // 先让几何法线朝向观察者，保证后续法线贴图 TBN 落在正确半球。
        float3 vdir = -WorldRayDirection();
        float3 n = normalize(mul(entityO2w, pr.normal.xyz));
        bool entering = dot(WorldRayDirection(), n) < 0.0;
        if (dot(n, vdir) < 0.0) {
            n = -n;
        }
        // 插值实体 UV 并采样对应图集，tint.rgb 是模型颜色乘数。
        ConstPtr<uint> eidxBuf = ConstPtr<uint>(g.idxAddr);
        ConstPtr<float2> euv = ConstPtr<float2>(g.uvAddr);
        uint e0 = eidxBuf[3u * pid + 0u];
        uint e1 = eidxBuf[3u * pid + 1u];
        uint e2 = eidxBuf[3u * pid + 2u];
        float3 ebary = float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
        float2 euvCoord = ebary.x * euv[e0] + ebary.y * euv[e1] + ebary.z * euv[e2];
        int texSlot = int(pr.tint.w + 0.5);
        float3 ep0 = mul(entityO2w, HitTriangleVertexPosition(0));
        float3 ep1 = mul(entityO2w, HitTriangleVertexPosition(1));
        float3 ep2 = mul(entityO2w, HitTriangleVertexPosition(2));
        float entityLod = rayConeTextureLod(rayCone, texSizePx(entityAlbedoTex[NonUniformResourceIndex(texSlot)]),
                ep0, ep1, ep2, euv[e0], euv[e1], euv[e2]);
        float4 entityTexel = entityAlbedoTex[NonUniformResourceIndex(texSlot)].SampleLevel(euvCoord, entityLod);
        MaterialHeader header = ConstPtr<MaterialHeader>(pc.materialTableAddr)[pr.materialId];
        uint material = header.model;
        float3 baseAlbedo = srgbToLinear(entityTexel.rgb) * pr.tint.rgb;
        float3 albedo = material == MATERIAL_GLASS
                ? lerp(float3(1.0, 1.0, 1.0), baseAlbedo, entityTexel.a) : baseAlbedo;
        Surface surface = evaluateMaterial(header, euvCoord, entityLod, albedo, n,
                ep0, ep1, ep2, euv[e0], euv[e1], euv[e2], vdir,
                header.params.x, header.params.y, pr.normal.w, material == MATERIAL_OPAQUE);
        n = surface.normal;

        payload.albedo = albedo;
        payload.normal = n;
        payload.hitT = RayTCurrent();
        // 实体运动矢量：有逐顶点位移就按重心坐标插值，否则使用刚体位移。
        if (g.dispAddr != 0) {
            ConstPtr<float4> dd = ConstPtr<float4>(g.dispAddr);
            payload.motionPrev = ebary.x * dd[e0].xyz + ebary.y * dd[e1].xyz + ebary.z * dd[e2].xyz;
        } else {
            payload.motionPrev = g.rigidDisp.xyz;
        }
        payload.f0 = surface.f0;
        float emission = material == MATERIAL_OPAQUE ? surface.emission : 0.0;
        float sss = material == MATERIAL_OPAQUE ? surface.sss : 0.0;
        payloadSetPacked(payload, material, surface.roughness, surface.metalness,
                emission, sss, header.params.z, header.params.w,
                materialEmissionSource(header, emission));
        if (material == MATERIAL_WATER && entering) {
            payload.flags |= PAYLOAD_WATER_ENTERING;
        }
        return;
    }

    Section sec = ConstPtr<Section>(pc.tableAddr)[InstanceID()]; // Section数组里的每个元素对应一个材质桶，InstanceID()就是桶索引
    // 地形 section 按材质桶组织，PrimitiveIndex 需加上当前桶的三角形基址。
    uint pid = PrimitiveIndex() + sec.triBase[GeometryIndex()];
    TerrainPrim pr = ConstPtr<TerrainPrim>(sec.primAddr)[pid];
    MaterialHeader materialHeader = ConstPtr<MaterialHeader>(pc.materialTableAddr)[pr.materialId];
    float3 n = normalize(pr.normal.xyz);
    float3 tint = pr.tint.rgb;

    // 水面几何按体积外侧绕序；翻转法线前先记录 entering，供 raygen 判断进入/离开水体。
    bool entering = dot(WorldRayDirection(), n) < 0.0;
    // 地形 UV 按三角形角点连续存储，命中后可直接用 pid 读取，减少分散索引访问。
    ConstPtr<float2> uvs = ConstPtr<float2>(sec.uvAddr); //一个由float2组成的数组，按三角形顶点连续存储UV坐标
    float2 uv0 = uvs[3u * pid + 0u];
    float2 uv1 = uvs[3u * pid + 1u];
    float2 uv2 = uvs[3u * pid + 2u];
    float3 bary = float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    float2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
    float3 tp0 = HitTriangleVertexPosition(0);
    float3 tp1 = HitTriangleVertexPosition(1);
    float3 tp2 = HitTriangleVertexPosition(2);
    float blockLod = rayConeTextureLod(rayCone, texSizePx(blockAlbedoAtlas), tp0, tp1, tp2, uv0, uv1, uv2);

    // 双面地形先翻转几何法线，再构建法线贴图 TBN，避免掠射角下扰动法线穿过表面。
    float3 vdir = -WorldRayDirection(); // 朝向观察者
    if (dot(n, vdir) < 0.0) {
        n = -n;
    }

    // 彩色玻璃/冰：payload.albedo 携带透射颜色，raygen 再按薄介质处理。
    if (materialHeader.model == MATERIAL_GLASS) {
        float4 gtex = blockAlbedoAtlas.SampleLevel(uv, blockLod);
        float3 gtexRgb = srgbToLinear(gtex.rgb);
        float3 glassAlbedo = lerp(float3(1.0, 1.0, 1.0), gtexRgb * tint, gtex.a);
        Surface glassSurface = evaluateMaterial(materialHeader, uv, blockLod, glassAlbedo, n,
                tp0, tp1, tp2, uv0, uv1, uv2, vdir,
                materialHeader.params.x, materialHeader.params.y, 0.0, false);
        n = glassSurface.normal;
        // 透明方块也应用破坏覆盖层，复用已采样的玻璃纹理。
        payload.albedo = applyBreaking(glassAlbedo,
                rayCone, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), n);
        payload.normal = n;
        payload.hitT = RayTCurrent();
        payload.motionPrev = float3(0.0, 0.0, 0.0);
        payload.f0 = glassSurface.f0;
        payloadSetPacked(payload, MATERIAL_GLASS, glassSurface.roughness,
                glassSurface.metalness, 0.0, 0.0, materialHeader.params.z,
                materialHeader.params.w, EMISSION_SOURCE_NONE);
        return;
    }

    // 水面只把生物群系水色写入 payload；不透明地形则采样方块反照率。
    payload.albedo = materialHeader.model == MATERIAL_WATER
            ? tint : srgbToLinear(blockAlbedoAtlas.SampleLevel(uv, blockLod).rgb) * tint;
    payload.hitT = RayTCurrent();
    payload.motionPrev = float3(0.0, 0.0, 0.0); // 静态地形：仅相机运动矢量
    uint material = materialHeader.model == MATERIAL_WATER ? MATERIAL_WATER : MATERIAL_OPAQUE;
    // normal.w 同时携带方块光照和非实心标记，用于发光与薄表面透射。
    float ew = pr.normal.w;
    bool nonSolid = ew >= 1.5;
    float emission = nonSolid ? ew - 2.0 : ew; // 启发式发光来源（方块光照等级）
    Surface surface = evaluateMaterial(materialHeader, uv, blockLod, payload.albedo, n,
            tp0, tp1, tp2, uv0, uv1, uv2, vdir,
            materialHeader.params.x, materialHeader.params.y, emission, nonSolid);
    n = surface.normal;
    // 非水地形应用破坏覆盖层；水体不可破坏，直接跳过。
    if (material != MATERIAL_WATER) {
        payload.albedo = applyBreaking(payload.albedo, rayCone, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), n);
    }
    payload.normal = n;
    payload.f0 = surface.f0;
    payloadSetPacked(payload, material, surface.roughness, surface.metalness,
            surface.emission, surface.sss, materialHeader.params.z, materialHeader.params.w,
            materialEmissionSource(materialHeader, surface.emission));
    if (material == MATERIAL_WATER && entering) {
        payload.flags |= PAYLOAD_WATER_ENTERING;
    }
}
