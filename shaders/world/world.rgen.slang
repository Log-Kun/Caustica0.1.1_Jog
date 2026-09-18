// 世界光追 raygen：重建主光线，迭代追踪路径，并写入降噪引导缓冲。
// 天空本身由 world.rmiss 计算；这里只负责把 miss 的天空辐射累加到路径中。
import world_common;

[[vk::push_constant]] WorldPushConstants pc;

// 光追加速结构和输出/引导纹理绑定。
[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;

[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture2D<float4> outImage;

[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> gNormal;
[[vk::binding(4, 0)]] [format("rgba16f")] RWTexture2D<float4> gAlbedo;
[[vk::binding(5, 0)]] [format("r32f")]    RWTexture2D<float>  gDepth;
[[vk::binding(6, 0)]] [format("rg16f")]   RWTexture2D<float2> gMotion;
[[vk::binding(7, 0)]] [format("rgba16f")] RWTexture2D<float4> gSpecAlbedo;
[[vk::binding(8, 0)]] [format("rg16f")]   RWTexture2D<float2> gSpecMotion;

// 每帧推送数据：相机、太阳、水体和材质开关等全局状态。
static WorldPush worldPush;

// 路径与 closest-hit/miss 之间共用的 payload。
static Payload payload;

// 阴影查询返回的可见性：rgb 是透射率，a 记录最近的水面交点 t。
static float4 shadowVis;

// 首次命中引导数据，供 DLSS-RR/降噪和运动向量使用。
static float3 gv_normal;
static float3 gv_albedo;
static float3 gv_specAlb;
static float gv_rough;
static float gv_emission;
static uint gv_emissionSource;

static float gv_depth;
static float3 gv_hitCamRel;
static float3 gv_motionHitCamRel;
static bool gv_motionUseRefracted;

static float3 gv_motionObjDisp;

// payload 字段访问宏
#define PAYLOAD_MATERIAL        (payload.flags & PAYLOAD_MATERIAL_MASK)                                      // 当前命中的材质类型
#define PAYLOAD_EMISSION_SOURCE ((payload.flags >> PAYLOAD_EMISSION_SOURCE_SHIFT) & PAYLOAD_EMISSION_SOURCE_MASK) // 发光来源标记
#define PAYLOAD_WATER_ENTERING  ((payload.flags & PAYLOAD_WATER_ENTERING) != 0u)                             // 光线是否正在进入水体
#define PAYLOAD_ROUGHNESS       (unpackHalf2(payload.roughMetal).x)                                           // 材质粗糙度
#define PAYLOAD_METALNESS       (unpackHalf2(payload.roughMetal).y)                                           // 材质金属度
#define PAYLOAD_EMISSION        (unpackHalf2(payload.emissionSss).x)                                          // 自发光强度
#define PAYLOAD_SSS             (unpackHalf2(payload.emissionSss).y)                                          // 薄表面透射强度
#define PAYLOAD_IOR             (unpackHalf2(payload.iorTransmission).x)                                      // 折射率
#define PAYLOAD_TRANSMISSION    (unpackHalf2(payload.iorTransmission).y)                                      // 透射比例
// 写入 miss/closest-hit 需要的追踪状态：是否显示日月，以及 ray cone 信息。
#define PAYLOAD_SET_TRACE_STATE(showCelestial, rayConeWidth, rayConeSpread) \
    payload.flags = (showCelestial) ? PAYLOAD_SHOW_CELESTIAL : 0u; \
    payload.rayCone = packHalf2(float2((rayConeWidth), (rayConeSpread)))

// 基础常量和主要材质参数。
#define PI 3.14159265359
#define INV_PI 0.31830988618
#define EMISSIVE_STRENGTH 5.
#define SURF_BIAS 0.005
#define RAY_TMIN 0.001

#define SKY_SPEC_ALBEDO float3(0.5, 0.5, 0.5)
#define SKY_DIFF_ALBEDO float3(0.5, 0.5, 0.5)

// 水体按平滑介电质处理，吸收使用 Beer-Lambert 消光。
#define WATER_IOR 1.333
#define WATER_DENSITY 0.05
#define WATER_ABSORB_FLOOR float3(0.015, 0.010, 0.008)

#define WATER_NIGHT_AMBIENT float3(0.002, 0.006, 0.014)
#define WATER_DAY_AMBIENT_SCALE 0.055
#define WATER_SUN_REFLECT_STRENGTH 100.
#define WATER_MOON_REFLECT_STRENGTH 80.

// 穿过玻璃时使用小偏移，避免越过与玻璃紧贴的邻近面。
#define GLASS_TRANSMIT_BIAS 1.0e-4

// 光锥跟踪像素覆盖范围，用于沿路径选择合适的纹理 mip。
#define RAY_CONE_MIN_SPREAD 1.0e-5
#define RAY_CONE_MIN_WIDTH 1.0e-5
#define RAY_CONE_DIFFUSE_SPREAD 0.25
#define RAY_CONE_GLOSSY_SPREAD_SCALE 0.35

// 简化的三维梯度噪声，供体积云密度场使用。
float hash13(float3 p3) {
    p3 = frac(p3 * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float3 grad3(float3 p) {
    float h = hash13(p);
    return normalize(frac(h * float3(17.13, 59.47, 15.23)) * 2. - 1.);
}

float noise3(float3 x) {
    float3 i = floor(x);
    float3 f = frac(x);
    float3 u = f * f * f * (f * (f * 6. - 15.) + 10.);
    float n000 = dot(grad3(i + float3(0.)), f - float3(0.));
    float n100 = dot(grad3(i + float3(1., 0., 0.)), f - float3(1., 0., 0.));
    float n010 = dot(grad3(i + float3(0., 1., 0.)), f - float3(0., 1., 0.));
    float n110 = dot(grad3(i + float3(1., 1., 0.)), f - float3(1., 1., 0.));
    float n001 = dot(grad3(i + float3(0., 0., 1.)), f - float3(0., 0., 1.));
    float n101 = dot(grad3(i + float3(1., 0., 1.)), f - float3(1., 0., 1.));
    float n011 = dot(grad3(i + float3(0., 1., 1.)), f - float3(0., 1., 1.));
    float n111 = dot(grad3(i + float3(1., 1., 1.)), f - float3(1., 1., 1.));
    float nx00 = lerp(n000, n100, u.x);
    float nx10 = lerp(n010, n110, u.x);
    float nx01 = lerp(n001, n101, u.x);
    float nx11 = lerp(n011, n111, u.x);
    return clamp(0.5 + lerp(lerp(nx00, nx10, u.y), lerp(nx01, nx11, u.y), u.z) * 0.85, 0., 1.);
}

// 体积云密度：先做高度包络，再用3D旋转分形噪声生成云体轮廓。
float cloudDensity(float3 p) {
    float worldY = p.y + worldPush.waterAnchor.z;
    float height = smoothstep(195., 238., worldY) * (1. - smoothstep(382., 430., worldY));
    float3 wp = float3((p.x + worldPush.waterAnchor.x) * 2., worldY, (p.z + worldPush.waterAnchor.y) * 2.);
    
    float shape = 0.;
    float amp = 0.58;
    float norm = 0.;
    float3 fp = ( wp + worldPush.waterParams.w * float3(2.1, 0.07, -1.15) ) * float3(0.0023, 0.0042, 0.0023); //worldPush.waterParams.w 是时间，控制云体的整体移动
    float3x3 octaveRot = float3x3(
             0.78,  0.16, -0.60,
            -0.10,  0.98,  0.18,
             0.62, -0.08,  0.78);
    for (int i = 0; i < 3; i++) {
        shape += noise3(fp) * amp;
        norm += amp;
        fp = mul(octaveRot, fp) * 6.17 + float3(19.1, 7.7, 31.3);
        amp *= 0.5;
    }
    shape /= norm;

    return smoothstep(0.6, 0.65, shape + height * 0.1) * height * 0.24;
}

// 水面高度场和法线：几何仍是平面，只扰动着色法线。
float WH(float2 position, float t) {
    float wavePhaseShift = length(position) * 0.1;
    float iter = 0.;
    float frequency = 0.4;
    float timeMultiplier = 2.;
    float weight = 1.;
    float sumOfValues = 0.;
    float sumOfWeights = 0.;
    for (int i = 0; i < 8; i++) {
        float2 dir = float2(sin(iter), cos(iter));
        float x = dot(dir, position) * frequency + t * timeMultiplier + wavePhaseShift;
        float wave = exp(sin(x) - 1.);
        position += dir * (-wave * cos(x)) * weight;
        sumOfValues += wave * weight;
        sumOfWeights += weight;
        weight *= 0.6;
        frequency *= 1.6;
        timeMultiplier *= -1.07;
        iter += 1232.399963;
    }
    return sumOfValues / sumOfWeights - 0.5;
}

float3 get_water_normal(float3 nGeo, float2 worldXZ, float t) {
    float eps = 0.05;
    float hL = WH(worldXZ - float2(eps, 0.), t);
    float hR = WH(worldXZ + float2(eps, 0.), t);
    float hD = WH(worldXZ - float2(0., eps), t);
    float hU = WH(worldXZ + float2(0., eps), t);
    float2 grad = float2(hR - hL, hU - hD) / (2. * eps);
    float3 up = normalize(float3(-grad.x, 1., -grad.y));
    return nGeo.y >= 0. ? up : -up;
}

// 根据水体颜色生成逐通道消光：深水通过衰减自然偏蓝/偏绿。
float3 waterExtinction(float3 tint) {
    return WATER_ABSORB_FLOOR + WATER_DENSITY * (float3(1., 1., 1.) - clamp(tint, 0., 1.));
}

// 水下焦散：对波面到接收平面的折射映射做有限差分。
#define CAUSTIC_EPS 0.25
#define CAUSTIC_MAX 6.
float2 causticLanding(float2 xz, float t, float3 inc, float h) {
    float3 nw = get_water_normal(float3(0., 1., 0.), xz, t);
    float3 rdw = refract(inc, nw, 1. / WATER_IOR);
    return xz + rdw.xz * (h / -rdw.y);
}

float waterCaustic(float3 exitPos, float3 lightDir, float waterDist) {

    float fade = smoothstep(0.06, 0.18, lightDir.y);
    if (fade <= 0.) return 1.;
    float h = waterDist * lightDir.y;
    float t = worldPush.waterParams.w;
    float3 inc = -lightDir;
    float2 base = exitPos.xz + worldPush.waterAnchor.xy;
    float2 p0 = causticLanding(base, t, inc, h);
    float2 px = causticLanding(base + float2(CAUSTIC_EPS, 0.), t, inc, h);
    float2 pz = causticLanding(base + float2(0., CAUSTIC_EPS), t, inc, h);
    float det = abs((px.x - p0.x) * (pz.y - p0.y) - (px.y - p0.y) * (pz.x - p0.x));

    float focus = (CAUSTIC_EPS * CAUSTIC_EPS) / det;
    return lerp(1., min(focus, CAUSTIC_MAX), fade);
}

// PBR 表面所需的 GGX、Fresnel 和辅助函数。
float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

float ggxD(float ndh, float rough) {
    float a = rough * rough;
    float a2 = a * a;
    float d = ndh * ndh * (a2 - 1.) + 1.;
    return a2 / (PI * d * d);
}

float ggxG1(float ndx, float rough) {
    float a = rough * rough;
    float a2 = a * a;
    return 2. * ndx / (ndx + sqrt(a2 + (1. - a2) * ndx * ndx));
}

float3 fresnelSchlick(float cosT, float3 f0) {
    float m = clamp(1. - cosT, 0., 1.);
    float m2 = m * m;
    return f0 + (1. - f0) * (m2 * m2 * m);
}

// 薄表面透射散射，主要用于树叶、草等薄片材质的背光效果。
#define SSS_G 0.6
#define SSS_STRENGTH 1.
#define MAX_SSS_BOUNCE 1
float hg(float cosT, float g) {
    float g2 = g * g;
    return (1. - g2) / (4. * PI * pow(max(0., 1. + g2 - 2. * g * cosT), 1.5));
}

// 近似积分 GGX 反射能量，写入 RR 镜面反照率引导。
float3 rrSpecularAlbedo(float3 specularColor, float alpha, float NoV) {
    NoV = abs(NoV);
    float NoV2 = NoV * NoV;
    float alpha2 = alpha * alpha;
    float4 X = float4(1., NoV, NoV2, NoV * NoV2);
    float4 Y = float4(1., alpha, alpha2, alpha * alpha2);

    float2 m1 = float2(
            dot(float2(0.99044, -1.28514), X.xy),
            dot(float2(1.29678, -0.755907), X.xy));
    float3 Xxyw = float3(X.x, X.y, X.w);
    float3 m2 = float3(
            dot(float3(1., 2.92338, 59.4188), Xxyw),
            dot(float3(20.3225, -27.0302, 222.592), Xxyw),
            dot(float3(121.563, 626.13, 316.627), Xxyw));
    float bias = dot(m1, Y.xy) / dot(m2, float3(Y.x, Y.y, Y.w));

    float2 m3 = float2(
            dot(float2(0.0365463, 3.32707), X.xy),
            dot(float2(9.0632, -9.04756), X.xy));
    float3 Xxzw = float3(X.x, X.z, X.w);
    float3 m4 = float3(
            dot(float3(1., 3.59685, -1.36772), Xxzw),
            dot(float3(9.04401, -16.3174, 9.22949), Xxzw),
            dot(float3(5.56589, 19.7886, -20.2123), Xxzw));
    float scale = dot(m3, Y.xy) / dot(m4, float3(Y.x, Y.y, Y.w));

    bias *= clamp(specularColor.g * 50., 0., 1.);
    return specularColor * max(0., scale) + float3(max(0., bias), max(0., bias), max(0., bias));
}

// 介电体 Fresnel，用于水/玻璃的反射和折射概率。
float fresnelDielectric(float cosI, float etaI, float etaT) {
    float sinT2 = (etaI * etaI) / (etaT * etaT) * max(0., 1. - cosI * cosI);
    if (sinT2 >= 1.) {
        return 1.;
    }
    float cosT = sqrt(1. - sinT2);
    float rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
    float rp = (etaI * cosT - etaT * cosI) / (etaI * cosT + etaT * cosI);
    return 0.5 * (rs * rs + rp * rp);
}

// 路径采样使用的简单伪随机数。
uint pcg(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float rndf(inout uint s) { return float(pcg(s)) * (1. / 4294967296.); }

// 主光线与光锥：由投影矩阵反推像素方向和像素角宽。
float3 primaryRayDir(float2 ndc) {
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1., 1.));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0., 1.));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    return normalize(farP - nearP);
}

float primaryRayConeSpread(float2 ndc, float2 size, float3 dir) {
    float2 onePixelNdc = 2. / size;
    float3 dx = primaryRayDir(ndc + float2(onePixelNdc.x, 0.));
    float3 dy = primaryRayDir(ndc + float2(0., onePixelNdc.y));
    return max(max(length(cross(dir, dx)), length(cross(dir, dy))), RAY_CONE_MIN_SPREAD);
}

// 常用采样函数：漫反射余弦采样、天体圆盘采样和 GGX VNDF。
float3 cosineDir(float3 n, inout uint s) {
    float u1 = rndf(s);
    float u2 = rndf(s);
    float r = sqrt(u1);
    float phi = 6.2831853 * u2;
    float3 t = normalize(abs(n.x) > 0.9 ? float3(0., 1., 0.) : float3(1., 0., 0.));
    float3 b = normalize(cross(n, t));
    t = cross(b, n);
    float3 local = float3(r * cos(phi), r * sin(phi), sqrt(max(0., 1. - u1)));
    return normalize(local.x * t + local.y * b + local.z * n);
}

float3 sampleDisc(float3 axis, float halfAngle, inout uint s) {
    float3 right = normalize(cross(axis, worldPush.celestial.xyz));
    float3 up = cross(right, axis);
    float t = tan(halfAngle);
    float r = sqrt(rndf(s)) * t;
    float phi = rndf(s) * 6.2831853;
    float u = r * cos(phi);
    float v = r * sin(phi);
    return normalize(axis + u * right + v * up);
}

float3 sampleGGXVNDF(float3 n, float3 ve, float rough, inout uint s) {
    float3 t = normalize(abs(n.x) > 0.9 ? float3(0., 1., 0.) : float3(1., 0., 0.));
    float3 b = normalize(cross(n, t));
    t = cross(b, n);
    float3 Ve = float3(dot(ve, t), dot(ve, b), dot(ve, n));
    float a = rough * rough;
    float3 Vh = normalize(float3(a * Ve.x, a * Ve.y, Ve.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0. ? float3(-Vh.y, Vh.x, 0.) * rsqrt(lensq) : float3(1., 0., 0.);
    float3 T2 = cross(Vh, T1);
    float u1 = rndf(s);
    float u2 = rndf(s);
    float r = sqrt(u1);
    float phi = 6.2831853 * u2;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float ss = 0.5 * (1. + Vh.z);
    p2 = (1. - ss) * sqrt(1. - p1 * p1) + ss * p2;
    float3 Nh = p1 * T1 + p2 * T2 + sqrt(max(0., 1. - p1 * p1 - p2 * p2)) * Vh;
    float3 Ne = normalize(float3(a * Nh.x, a * Nh.y, max(0., Nh.z)));
    return normalize(Ne.x * t + Ne.y * b + Ne.z * n);
}

// 光追 SBT 和剔除标志。
#define CULL_SECONDARY 0x01u
#define CULL_PRIMARY 0x02u
#define TERRAIN_BUCKETS 4u
#define SBT_RADIANCE 0u
#define SBT_SHADOW TERRAIN_BUCKETS
#define SBT_STRIDE_BUCKET 1u
RayDesc makeRay(float3 origin, float tmin, float3 dir) {
    RayDesc r;
    r.Origin = origin;
    r.TMin = tmin;
    r.Direction = dir;
    r.TMax = 256.; // 阴影追踪距离限制256格
    return r;
}

// 光追主函数：重排线程后调用 closest-hit/miss shader。
void traceRadianceReordered(uint cullMask, float3 ro, float tmin, float3 rd) {
/*
topLevelAS          场景加速结构，里面有所有可光追的物体
RAY_FLAG_NONE       普通 radiance ray，不跳过 closest-hit
cullMask            控制这条 ray 能看见哪些实例
SBT_RADIANCE        用 radiance 那组 shader binding table
SBT_STRIDE_BUCKET   SBT 里不同材质 bucket 的跨度
0u                  miss shader index
makeRay(...)        这条 ray 的 origin/tmin/direction
payload             要传给 hit/miss shader 的数据包
*/
    HitObject hObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_NONE, cullMask, SBT_RADIANCE, SBT_STRIDE_BUCKET, 0u, makeRay(ro, tmin, rd), payload);
    
    ReorderThread(hObj); // 线程重排，仅优化性能用（SER优化）

    HitObject::Invoke(topLevelAS, hObj, payload);
/*
如果 hObj 是命中几何：
    调用 world.rchit.slang 的 closesthit
    rchit 写 payload.albedo / normal / hitT / roughness ...
如果 hObj 是 miss：
    调用 world.rmiss.slang 的 miss
    miss 写 payload.albedo = 天空颜色，payload.hitT = -1
*/
}

// 可见性查询：只跑 any-hit/透射逻辑，返回阴影透过率。
float3 visibility(float3 origin, float3 dir) {
    shadowVis = float4(1., 1., 1., -1.);
    HitObject hObj = HitObject::TraceRay(topLevelAS,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
            CULL_SECONDARY, SBT_SHADOW, SBT_STRIDE_BUCKET, 0u,
            makeRay(origin, RAY_TMIN, dir), shadowVis);
    return hObj.IsMiss() ? shadowVis.rgb : float3(0.);
}

float3 waterCelestialHighlight(float3 hitPos, float3 n, float3 viewDir, float3 bodyDir, float3 radiance, float F, float strength, float sharpness, float bias, float weight) {
    float bodyUp = clamp((bodyDir.y + 0.0625) / 0.125, 0., 1.);
    float3 halfDir = normalize(bodyDir + viewDir);
    float term = clamp(dot(n, halfDir) * sharpness - bias, 0., 1.);
    float mask = term * term;
    mask *= mask;
    if (mask <= 0. || bodyUp <= 0. || weight <= 0.) {
        return float3(0.);
    }
    return radiance * F * mask * visibility(hitPos + n * SURF_BIAS, bodyDir) * bodyUp * weight * strength;
}

// 运动向量辅助，用上一帧矩阵投影引导点。
float2 projectPrevNdc(float3 worldPos) {
    float4 clip = mul(worldPush.prevViewProj, float4(worldPos - worldPush.camOffset + worldPush.camDelta, 1.));
    return clip.xy / clip.w;
}

float2 previousReflectionNdc(float3 surfacePos, float3 surfaceNormal, float3 reflectedWorldPos, float3 reflectedMotionPrev) {
    float3 n = normalize(surfaceNormal);
    float3 prevHit = reflectedWorldPos - reflectedMotionPrev;
    float3 mirroredHit = prevHit - 2. * n * dot(prevHit - surfacePos, n);
    return projectPrevNdc(mirroredHit);
}

float2 specularReflectionMotion(float3 surfacePos, float3 primaryDir, float2 currentNdc, float2 size, float primaryConeSpread) {
    if (length(gv_normal) < 0.5 || max(gv_specAlb.r, max(gv_specAlb.g, gv_specAlb.b)) <= 0.001 || gv_rough > 0.5) {
        return float2(0.);
    }

    float3 n = normalize(gv_normal);
    float3 specDir = reflect(primaryDir, n);

    payload.hitT = -1.;
    PAYLOAD_SET_TRACE_STATE(false,
            max(length(surfacePos - worldPush.camOffset) * primaryConeSpread, RAY_CONE_MIN_WIDTH),
            max(primaryConeSpread, RAY_CONE_MIN_SPREAD));
    traceRadianceReordered(CULL_SECONDARY, surfacePos + n * SURF_BIAS, RAY_TMIN, specDir);

    float3 reflectedHit;
    float3 reflectedMotionPrev;
    if (payload.hitT > 0.) {
        reflectedHit = surfacePos + specDir * payload.hitT;

        reflectedMotionPrev = PAYLOAD_MATERIAL == MATERIAL_OPAQUE ? payload.motionPrev : float3(0.);
    } else {
        reflectedHit = surfacePos + specDir * 1.0e6;
        reflectedMotionPrev = float3(0.);
    }
    float2 prevNdc = previousReflectionNdc(surfacePos, n, reflectedHit, reflectedMotionPrev);
    return (prevNdc - currentNdc) * 0.5 * size;
}

// 水面折射引导：找到水面后方真正可见的物体，用它来稳定透过水的运动向量。
void refractedGuideHit(float3 surfacePos, float3 incidentDir, float3 surfaceNormal, bool inWaterAtSurface,
                       float materialIor,
                       float rayConeWidth, float rayConeSpread,
                       out float3 hitCamRel, out float3 motionPrev, out bool refracted, out float3 diffuseAlbedo,
                       out float3 specAlbedo) {
    hitCamRel = surfacePos - worldPush.camOffset;
    motionPrev = float3(0.);
    refracted = false;
    diffuseAlbedo = float3(0.);
    specAlbedo = float3(0.);

    float etaI = inWaterAtSurface ? materialIor : 1.;
    float etaT = inWaterAtSurface ? 1. : materialIor;
    float3 refractedDir = refract(incidentDir, surfaceNormal, etaI / etaT);
    if (dot(refractedDir, refractedDir) <= 0.) {
        return;
    }
    refractedDir = normalize(refractedDir);
    refracted = true;

    float3 ro = surfacePos - surfaceNormal * SURF_BIAS;
    payload.hitT = -1.;
    PAYLOAD_SET_TRACE_STATE(false, rayConeWidth, rayConeSpread);
    traceRadianceReordered(CULL_SECONDARY, ro, RAY_TMIN, refractedDir);
    if (payload.hitT < 0.) {
        hitCamRel = (ro + refractedDir * 1.0e6) - worldPush.camOffset;
        motionPrev = float3(0.);
        diffuseAlbedo = SKY_DIFF_ALBEDO;
        specAlbedo = SKY_SPEC_ALBEDO;
        return;
    }

    float3 hitPos = ro + refractedDir * payload.hitT;
    hitCamRel = hitPos - worldPush.camOffset;
    uint material = PAYLOAD_MATERIAL;
    motionPrev = material != MATERIAL_WATER ? payload.motionPrev : float3(0.);
    if (material == MATERIAL_OPAQUE) {
        bool pbr = (worldPush.flags & 2u) != 0u;
        diffuseAlbedo = pbr ? payload.albedo * (1. - clamp(PAYLOAD_METALNESS, 0., 1.)) : payload.albedo;
        if (pbr) {
            float3 f0 = payload.f0;
            float rough = clamp(PAYLOAD_ROUGHNESS, 0., 1.);
            float NoV = clamp(dot(payload.normal, -refractedDir), 0., 1.);
            specAlbedo = rrSpecularAlbedo(f0, rough * rough, NoV);
        }
    }
}







// 主路径积分循环：处理天空、体积、水、玻璃、粒子和常规不透明表面。
float3 tracePath(float3 ro, float3 rd, float primaryConeSpread, inout uint seed) {
    float3 diffuse = float3(0.);
    float3 throughput = float3(1.); //剩余透射率
    float rayConeWidth = 0.;
    float rayConeSpread = max(primaryConeSpread, RAY_CONE_MIN_SPREAD);
    int maxBounces = int(worldPush.maxBounces);
    int rrStart = maxBounces <= 3 ? 1 : 2;
    bool pbr = (worldPush.flags & 2u) != 0u;

    bool inWater = (worldPush.flags & 1u) != 0u;

    float3 waterExt = waterExtinction(worldPush.waterParams.xyz);
    bool waterWaves = (worldPush.flags & 16u) != 0u;

    bool showCelestial = true; // 反射是否包含日月 防止漫反射出现噪点

    for (int bounce = 0; bounce <= maxBounces; bounce++) {
        // 先追踪几何，后面的体积和表面分支都以这次命中距离为上限。
        payload.hitT = -1.;
        PAYLOAD_SET_TRACE_STATE(showCelestial, rayConeWidth, rayConeSpread);
        traceRadianceReordered(bounce == 0 ? CULL_PRIMARY : CULL_SECONDARY, ro, 0., rd);

        if (!inWater) {
            {   // 体积光部分 近地薄雾：只处理空气中的一小段，水下衰减交给水体吸收。
                // 近地体积雾参数，与云层是两套独立积分。
                #define sigmaT 0.012
                #define VOLUME_PHASE_G 0.72
                #define VOLUME_SCATTER_ALBEDO_TINT float3(0.92, 0.97, 1.)*0.65
                #define VOLUME_STEP_LENGTH 0.4
                //#define VOLUME_STEP_GROW_START 1.
                //#define VOLUME_STEP_GROW_END 60.
                //#define VOLUME_STEP_MAX_SCALE 64.
                #define VOLUME_SKY_LAYER_HEIGHT 24. // 天空最多步进次数 这部分主要是积分
                #define VOLUME_LIGHT_DISTANCE 32.
                float volumeLimit=64.;

                if (payload.hitT < 0. && rd.y > 0.)volumeLimit = min(volumeLimit, VOLUME_SKY_LAYER_HEIGHT / max(rd.y, 0.05));

                float segmentLength = payload.hitT >= 0.
                    ? min(payload.hitT, volumeLimit)
                    : volumeLimit;

                float3 volumeLightDir = normalize(worldPush.lightDir.xyz);
                float phase = hg(dot(volumeLightDir, rd), VOLUME_PHASE_G);
                float lightTransmittance = exp(-sigmaT * VOLUME_LIGHT_DISTANCE);
                float viewTransmittance = 1.;
                float volumeStepJitter = rndf(seed) * 0.3 + 0.6;

                float t = 0.;
                while (t < segmentLength) {
                    float volumeStepLen = VOLUME_STEP_LENGTH * volumeStepJitter * (1.+t);
                    float stepLen = min(volumeStepLen, segmentLength - t);
                    float3 volumePos = ro + rd * (t + 0.5 * stepLen);
                    float3 segmentScatter = VOLUME_SCATTER_ALBEDO_TINT * (1. - exp(-sigmaT * stepLen));

                    diffuse += throughput
                            * viewTransmittance
                            * segmentScatter
                            * phase
                            * worldPush.lightRadiance.xyz
                            * visibility(volumePos, volumeLightDir)
                            * lightTransmittance;

                    viewTransmittance *= exp(-sigmaT * stepLen);
                            t += stepLen;
                }
                
                throughput *= exp(-sigmaT * segmentLength);
            } // 体积光结束



            { // 云层部分 求出当前光线穿过云盒的 t 范围，再在这个范围内做稀疏采样。
                float cloudBase = 190.;
                float cloudTop = 430.;
                float cloudFadeStart = 1400.;
                float cloudMaxDistance = 1800.;
                float rayLimit = payload.hitT >= 0. ? min(payload.hitT, cloudMaxDistance) : cloudMaxDistance;

                float t0 = 0.;
                float t1 = -1.;
                if (rayLimit > 0.) {
                    float roWorldY = ro.y + worldPush.waterAnchor.z;
                    if (abs(rd.y) > 0.01) {
                        float ta = (cloudBase - roWorldY) / rd.y;
                        float tb = (cloudTop - roWorldY) / rd.y;
                        t0 = max(min(ta, tb), 0.);
                        t1 = min(max(ta, tb), rayLimit);
                    } else if (roWorldY > cloudBase && roWorldY < cloudTop) {
                        t1 = rayLimit;
                    }
                }

                if (t1 > t0) {
                    float T = 1.;
                    float3 cloudL = float3(0.);
                    float sigmaT_water = 0.012;
                    float scatterAlbedo = 0.88;
                    float cloudG = 0.38;
                    float3 cloudLightDir = normalize(worldPush.lightDir.xyz);
                    float cloudLightUp = smoothstep(0.02, 0.15, cloudLightDir.y);
                    float hgPhase = hg(dot(rd, cloudLightDir), cloudG);
                    float cloudStepLen = (t1 - t0) / 8.;
                    float jitter = rndf(seed) * 0.8 + 0.1;
                    float day = worldPush.sunDir.w;

                    for (int s = 0; s < 3; s++) {
                        float t = t0 + (float(s) + jitter) * cloudStepLen;
                        float3 cp = ro + rd * t;
                        float distFade = 1. - smoothstep(cloudFadeStart, cloudMaxDistance, t);
                        float rawDensity = cloudDensity(cp);
                        float density = rawDensity * distFade;
                        if (density <= 0.001) {
                            continue;
                        }

                        float lightT = 0.45;
                        if (cloudLightUp > 0.) {
                            float lightProbe = cloudDensity(cp + cloudLightDir * 90.);
                            float edgeLight = clamp((rawDensity - lightProbe) * 3. + 0.35, 0., 1.);
                            lightT = lerp(0.20, 1., edgeLight * edgeLight * (3. - 2. * edgeLight));
                        }

                        float alpha = 1. - exp(-sigmaT_water * density * cloudStepLen);
                        float viewLight = max(dot(rd, cloudLightDir), 0.);
                        float phaseFit = lerp(0.07, min(hgPhase * 1.8, 0.55), smoothstep(0., 0.9, viewLight));
                        float powder = 1. - exp(-density * 2.2);
                        float rim = smoothstep(0.15, 0.85, lightT) * smoothstep(0.20, 0.95, viewLight);
                        float3 ambientCloud = lerp(float3(0.010, 0.014, 0.028), float3(0.20, 0.22, 0.24), day);
                        float3 directCloud = worldPush.lightRadiance.xyz * scatterAlbedo * cloudLightUp
                                * (phaseFit * lightT * (0.22 + 0.78 * powder) + rim * 0.030) * day;

                        cloudL += T * alpha * (ambientCloud + directCloud);
                        T *= 1. - alpha;
                        if (T < 0.02)break;
                    }

                    diffuse += throughput * cloudL;
                    throughput *= T;
                }
            } // 云层结束
        
        }else{ // 水下体积光
                    
            #define WATER_VOLUME_MAX_DISTANCE 64.
            #define WATER_VOLUME_STEPS 8
            #define WATER_VOLUME_SCATTER 0.08
            #define WATER_VOLUME_PHASE_G 0.45
            #define WATER_VOLUME_INTENSITY 0.5

            float waterSegment = payload.hitT >= 0. ? payload.hitT : WATER_VOLUME_MAX_DISTANCE;
            waterSegment = min(waterSegment, WATER_VOLUME_MAX_DISTANCE);
            float waterStepLen = waterSegment / float(WATER_VOLUME_STEPS);
            float day = clamp(worldPush.sunDir.w, 0., 1.);
            float waterJitter = rndf(seed) * 0.8 + 0.1;
            float3 waterLightDir = normalize(worldPush.lightDir.xyz);
            float waterPhase = min(hg(dot(rd, waterLightDir), WATER_VOLUME_PHASE_G), 0.35);
            float3 sigmaT_water = max(waterExt, WATER_ABSORB_FLOOR);
            float3 waterVolumeTr = float3(1.);

            for (int s = 0; s < WATER_VOLUME_STEPS; s++) {
                float t = (float(s) + waterJitter) * waterStepLen;
                if (t >= waterSegment)break;
                float stepLen = min(waterStepLen, waterSegment - t);
                float3 p = ro + rd * (t + 0.5 * stepLen);
                float3 vis = visibility(p, waterLightDir);
                if (waterWaves && shadowVis.a > 0.01) {
                    vis *= waterCaustic(p + waterLightDir * shadowVis.a, waterLightDir, shadowVis.a);
                }
                float3 stepScatter = clamp(worldPush.waterParams.xyz, 0.15, 1.) * WATER_VOLUME_SCATTER * waterPhase * (1. - exp(-sigmaT_water * stepLen));
                diffuse += throughput * waterVolumeTr * stepScatter * worldPush.lightRadiance.xyz * vis * day * WATER_VOLUME_INTENSITY;
                waterVolumeTr *= exp(-sigmaT_water * stepLen);
            }

            float3 waterDayAmbient = (worldPush.waterParams.xyz + float3(0.02, 0.10, 0.14)) * WATER_DAY_AMBIENT_SCALE;
            float3 waterFogColor = lerp(WATER_NIGHT_AMBIENT, waterDayAmbient, day);
            diffuse += throughput * waterFogColor * (float3(1.) - waterVolumeTr);
            throughput *= waterVolumeTr;
        }

        // 未命中几何体：累积 miss shader 给出的天空，并结束当前路径。
        if (payload.hitT < 0.) {
            float3 sky = payload.albedo;
            if (bounce == 0) {
                gv_normal = float3(0.);
                gv_albedo = SKY_DIFF_ALBEDO;
                gv_specAlb = SKY_SPEC_ALBEDO;
                gv_rough = 1.;
                gv_emission = 0.;
                gv_emissionSource = EMISSION_SOURCE_NONE;
                gv_hitCamRel = rd * 1.0e6;
                gv_motionHitCamRel = gv_hitCamRel;
                gv_motionUseRefracted = false;
                gv_motionObjDisp = float3(0.);
            }
            diffuse += throughput * sky;
            break;
        }

        float3 n = payload.normal;
        float3 hitPos = ro + rd * payload.hitT;
        rayConeWidth = max(rayConeWidth + rayConeSpread * max(payload.hitT, 0.), RAY_CONE_MIN_WIDTH);
        uint material = PAYLOAD_MATERIAL;
        if (bounce == 0) {
            gv_emission = PAYLOAD_EMISSION;
            gv_emissionSource = PAYLOAD_EMISSION_SOURCE;
        }

        // 玻璃是薄介质表面：按 Fresnel 在反射和透射之间随机选择。
        if (material == MATERIAL_GLASS) {
            float3 glassTint = payload.albedo;
            float materialIor = max(PAYLOAD_IOR, 1.0e-3);
            float transmission = clamp(PAYLOAD_TRANSMISSION, 0., 1.);
            float cosI = clamp(dot(-rd, n), 0., 1.);
            float F = fresnelDielectric(cosI, 1., materialIor);
            if (bounce == 0) {
                gv_normal = n;
                gv_rough = clamp(PAYLOAD_ROUGHNESS, 0., 1.);
                gv_specAlb = float3(F, F, F);
                gv_albedo = glassTint;
                gv_hitCamRel = hitPos - worldPush.camOffset;
                gv_motionHitCamRel = gv_hitCamRel;
                gv_motionUseRefracted = false;
                gv_motionObjDisp = payload.motionPrev;
            }
            if (rndf(seed) < F) {
                rd = reflect(rd, n);
                ro = hitPos + n * SURF_BIAS;
            } else {
                throughput *= glassTint * transmission;
                ro = hitPos - n * GLASS_TRANSMIT_BIAS;
            }
            showCelestial = true;
            if (bounce >= rrStart) {
                float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0., 1.);
                if (rndf(seed) > q)break;
                throughput /= q;
            }
            continue;
        }

        // 粒子按近似双面薄片处理，可接收直射光并继续散射。
        if (material == MATERIAL_PARTICLE) {
            float3 albedo = payload.albedo;
            gv_normal = n;
            gv_albedo = albedo;
            gv_specAlb = float3(0.);
            gv_rough = 1.;
            gv_hitCamRel = hitPos - worldPush.camOffset;
            gv_motionHitCamRel = gv_hitCamRel;
            gv_motionUseRefracted = false;
            gv_motionObjDisp = payload.motionPrev;

            float3 lightDir = worldPush.lightDir.xyz;
            float lightHalfAngle = worldPush.lightDir.w;
            if (lightHalfAngle > 0.) {
                lightDir = sampleDisc(lightDir, lightHalfAngle, seed);
            }

            float signedNdl = dot(n, lightDir);
            float ndl = abs(signedNdl);
            if (ndl > 0.) {
                float3 shadowOrigin = hitPos + (signedNdl >= 0. ? n : -n) * SURF_BIAS;
                { float3 vis = visibility(shadowOrigin, lightDir); if (max(vis.r, max(vis.g, vis.b)) > 0.) diffuse += throughput * albedo * INV_PI * worldPush.lightRadiance.xyz * ndl * vis; }
            }

            if (bounce >= maxBounces) {
                break;
            }
            throughput *= albedo;
            ro = hitPos + n * SURF_BIAS;
            rd = cosineDir(n, seed);
            rayConeSpread = max(rayConeSpread, RAY_CONE_DIFFUSE_SPREAD);
            showCelestial = false;
            continue;
        }

        //水体渲染 水面是体积边界：在反射和折射之间选择，同时切换 inWater 状态。
        if (material == MATERIAL_WATER) {
            float materialIor = max(PAYLOAD_IOR, 1.0e-3);
            float transmission = clamp(PAYLOAD_TRANSMISSION, 0., 1.);

            bool materialEntering = PAYLOAD_WATER_ENTERING;

            if (waterWaves) {
                float2 waveXZ = hitPos.xz + worldPush.waterAnchor.xy + (abs(n.y) < 0.5 ? hitPos.y : 0.);
                n = get_water_normal(n, waveXZ, worldPush.waterParams.w);
            }
            waterExt = waterExtinction(payload.albedo);
            float cosI = clamp(dot(-rd, n), 0., 1.);
            float etaI = inWater ? materialIor : 1.;
            float etaT = inWater ? 1. : materialIor;
            float F = fresnelDielectric(cosI, etaI, etaT);
            if (bounce == 0) {
                gv_normal = n;
                gv_rough = clamp(PAYLOAD_ROUGHNESS, 0., 1.);
                gv_hitCamRel = hitPos - worldPush.camOffset;

                float3 refractedAlbedo;
                float3 refractedSpecAlbedo;
                refractedGuideHit(hitPos, rd, n, inWater, materialIor, rayConeWidth, rayConeSpread,
                        gv_motionHitCamRel, gv_motionObjDisp,
                        gv_motionUseRefracted, refractedAlbedo, refractedSpecAlbedo);
                gv_albedo = refractedAlbedo;
                gv_specAlb = refractedSpecAlbedo;

                //水面高光补偿部分
                float day = clamp(worldPush.sunDir.w, 0., 1.);
                float3 viewDir = normalize(-rd);
                diffuse += throughput * waterCelestialHighlight(hitPos, n, viewDir, normalize(worldPush.sunDir.xyz),
                        worldPush.lightRadiance.xyz, F, WATER_SUN_REFLECT_STRENGTH, 400., 399., day);
                diffuse += throughput * waterCelestialHighlight(hitPos, n, viewDir, normalize(worldPush.moonDir.xyz),
                        worldPush.lightRadiance.xyz * float3(0.55, 0.65, 1.), F, WATER_MOON_REFLECT_STRENGTH, 320., 319., 1. - day);
            }
            if (rndf(seed) < F) {
                rd = reflect(rd, n);
                ro = hitPos + n * SURF_BIAS;
            } else {
                rd = refract(rd, n, etaI / etaT);
                throughput *= transmission;
                ro = hitPos - n * SURF_BIAS;

                inWater = materialEntering;
            }

            showCelestial = true;
            if (bounce >= rrStart) {
                float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0., 1.);
                if (rndf(seed) > q)break;
                throughput /= q;
            }
            continue;
        }


        // 常规不透明表面：读取材质参数，计算直接光照和间接反射。
        float3 albedo = payload.albedo;
        float sss = PAYLOAD_SSS;
        float3 p = hitPos + n * SURF_BIAS;
        float3 v = -rd;

        float rough = pbr ? clamp(PAYLOAD_ROUGHNESS, 0., 1.) : 1.; //读取PBR参数
        float metal = pbr ? clamp(PAYLOAD_METALNESS, 0., 1.) : 0.;
        float3 diffAlb = albedo * (1. - metal);

        float3 F0 = pbr ? payload.f0 : lerp(float3(0.04, 0.04, 0.04), albedo, metal);

        if (bounce == 0) {
            gv_normal = n;
            gv_albedo = pbr ? diffAlb : albedo;
            gv_rough = rough;
            gv_specAlb = pbr ? rrSpecularAlbedo(F0, rough * rough, dot(n, v)) : float3(0.);
            gv_hitCamRel = ro + rd * payload.hitT - worldPush.camOffset;
            gv_motionHitCamRel = gv_hitCamRel;
            gv_motionUseRefracted = false;
            gv_motionObjDisp = payload.motionPrev;
        }

        float emission = PAYLOAD_EMISSION;
        // 发光表面直接向路径贡献辐射。
        if (emission > 0.001) diffuse += throughput * albedo * emission * EMISSIVE_STRENGTH;

        float3 lightDir = worldPush.lightDir.xyz;
        float lightHalfAngle = worldPush.lightDir.w;
        if (lightHalfAngle > 0.) lightDir = sampleDisc(lightDir, lightHalfAngle, seed);

        float ndl = max(0., dot(n, lightDir));
        if (ndl > 0.) {
            float3 vis = visibility(p, lightDir);

            if (inWater && waterWaves && shadowVis.a > 0.01) {
                vis *= waterCaustic(p + lightDir * shadowVis.a, lightDir, shadowVis.a);
            }
            if (max(vis.r, max(vis.g, vis.b)) > 0.01) {
                float3 brdf = diffAlb * INV_PI;
                if (pbr) {
                    float3 h = normalize(lightDir + v);
                    float ndh = max(0., dot(n, h));
                    float ndv = dot(n, v);
                    float vdh = max(0., dot(v, h));
                    float D = ggxD(ndh, rough);
                    float G = ggxG1(ndv, rough) * ggxG1(ndl, rough);
                    float3 F = fresnelSchlick(vdh, F0);
                    brdf += (D * G) * F / (4. * ndv * ndl);
                }
                diffuse += throughput * brdf * worldPush.lightRadiance.xyz * ndl * vis;
            }
        }

        // 薄表面背光透射，只在前几次反弹计算以控制开销。
        if (sss > 0.001 && bounce <= MAX_SSS_BOUNCE) {
            float backNdl = max(0., dot(-n, lightDir));
            if (backNdl > 0.01) {
                float3 visB = visibility(hitPos - n * SURF_BIAS, lightDir);

                if (inWater && waterWaves && shadowVis.a > 0.01) {
                    visB *= waterCaustic(hitPos + lightDir * shadowVis.a, lightDir, shadowVis.a);
                }
                if (max(visB.r, max(visB.g, visB.b)) > 0.01) {
                    float cosT = dot(lightDir, rd);
                    diffuse += throughput * diffAlb * sss * SSS_STRENGTH * hg(cosT, SSS_G) * backNdl * worldPush.lightRadiance.xyz * visB;
                }
            }
        }

        // 间接延续：PBR 模式在漫反射和镜面瓣之间按能量随机选择。
        if (pbr) {
            float ps = clamp(luminance(F0) / (luminance(F0) + luminance(diffAlb)), 0.1, 0.9);
            if (rndf(seed) < ps) { // 镜面反射
                float3 h = sampleGGXVNDF(n, v, rough, seed);
                float3 l = reflect(rd, h);
                float ndl2 = dot(n, l);
                if (ndl2 <= 0.01)break;

                float3 F = fresnelSchlick(max(0., dot(v, h)), F0);
                throughput *= F * ggxG1(ndl2, rough) / ps;
                ro = p;
                rd = l;
                rayConeSpread = max(rayConeSpread, rough * rough * RAY_CONE_GLOSSY_SPREAD_SCALE);
                showCelestial = true;
            } else { // 漫反射
                throughput *= diffAlb / (1. - ps);
                ro = p;
                rd = cosineDir(n, seed);
                rayConeSpread = max(rayConeSpread, RAY_CONE_DIFFUSE_SPREAD);
                showCelestial = false;
            }
        } else {
            throughput *= albedo;
            showCelestial = false;
            ro = p;
            rd = cosineDir(n, seed);
            rayConeSpread = max(rayConeSpread, RAY_CONE_DIFFUSE_SPREAD);
        }

        if (bounce >= rrStart) { // 当路径的剩余透射率过低时，随机终止路径以节省计算。
            float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0., 1.);
            if (rndf(seed) > q)break;
            throughput /= q;
        }
    }
    return diffuse;
}











// Ray generation 入口：为当前像素发射主光线，平均 SPP，并写出引导缓冲。
[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    int2 pix = int2(DispatchRaysIndex().xy);
    float2 size = float2(DispatchRaysDimensions().xy);
    float2 uv = (float2(pix) + 0.5) / size;
    // 主光线使用 DLSS-RR 提供的亚像素抖动；运动向量之后会抵消这部分抖动。
    float2 jndc = (uv + worldPush.jitter / size) * 2. - 1.;

    float3 origin,dir;
    {
        float4 nearH = mul(worldPush.invViewProj, float4(jndc.x, jndc.y, 1., 1.));
        float4 farH = mul(worldPush.invViewProj, float4(jndc.x, jndc.y, 0., 1.));
        float3 nearP = nearH.xyz / nearH.w;
        float3 farP = farH.xyz / farH.w;
        origin = nearP + worldPush.camOffset;
        dir = normalize(farP - nearP);
    }

    float rayConeSpread = primaryRayConeSpread(jndc, size, dir); //用于计算光线锥的初始扩散角度，影响路径追踪的采样精度。

    uint seed = (uint(pix.x) * 1973u + uint(pix.y) * 9277u + 26699u) ^ (worldPush.frameIndex * 2u);

    uint spp = max(worldPush.spp, 1u);
    float3 frameRadiance = float3(0.);
    for (uint s = 0u; s < spp; s++) { //核心循环
        seed = pcg(seed);
        frameRadiance += tracePath(origin, dir, rayConeSpread, seed);
    }

    outImage[pix] = float4(frameRadiance/float(spp), 1.);




    float4 curClip = mul(worldPush.curViewProj, float4(gv_hitCamRel, 1.));
    gv_depth = curClip.w > 0. ? curClip.z / curClip.w : 0.;

    gNormal[pix] = float4(gv_normal, gv_rough);
    gAlbedo[pix] = float4(gv_albedo, 1.);
    gDepth[pix] = gv_depth;
    gSpecAlbedo[pix] = float4(gv_specAlb, 1.);

    float4 prevClip = mul(worldPush.prevViewProj, float4(gv_motionHitCamRel + worldPush.camDelta - gv_motionObjDisp, 1.));
    float2 prevNdc = prevClip.xy / prevClip.w;
    float2 curNdc = jndc;

    // 水面像素的运动向量跟踪折射后的命中点，而不是水面自身。
    if (gv_motionUseRefracted) {
        float4 curClipRefr = mul(worldPush.curViewProj, float4(gv_motionHitCamRel, 1.));
        curNdc = curClipRefr.xy / curClipRefr.w;
    }

    float2 motion = (prevNdc - curNdc) * 0.5 * size;
    gMotion[pix] = motion;

    float2 specMotion = specularReflectionMotion(gv_hitCamRel + worldPush.camOffset, dir, jndc, size, rayConeSpread);
    gSpecMotion[pix] = specMotion;

}
