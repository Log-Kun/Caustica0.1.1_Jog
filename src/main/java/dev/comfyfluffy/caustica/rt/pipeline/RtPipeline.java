package dev.comfyfluffy.caustica.rt.pipeline;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VK12;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBindingFlagsCreateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkRayTracingPipelineCreateInfoKHR;
import org.lwjgl.vulkan.VkRayTracingShaderGroupCreateInfoKHR;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkStridedDeviceAddressRegionKHR;
import org.lwjgl.vulkan.VkWriteDescriptorSet;
import org.lwjgl.vulkan.VkWriteDescriptorSetAccelerationStructureKHR;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.RtDeviceBringup;
import dev.comfyfluffy.caustica.rt.accel.RtAccel;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static org.lwjgl.vulkan.EXTOpacityMicromap.VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_SHADER_STAGE_MISS_BIT_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_SHADER_STAGE_RAYGEN_BIT_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_SHADER_UNUSED_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.vkCmdTraceRaysKHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.vkCreateRayTracingPipelinesKHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.vkGetRayTracingShaderGroupHandlesKHR;

/**
 * An RT pipeline with an SBT of {raygen + N miss + triangle hit groups} and a descriptor
 * set of {binding 0 = TLAS, binding 1 = storage image}. Built from SPIR-V resources. Update the
 * bindings with {@link #setTlas}/{@link #setStorageImage}, then {@link #trace}. Reusable across
 * the triangle spike and terrain (extend the descriptor layout there as needed). Multiple miss
 * shaders (e.g. a primary sky miss at index 0 plus a shadow/visibility miss at index 1) are
 * supported by passing an array; {@code traceRayEXT}'s {@code missIndex} selects among them.
 */
public final class RtPipeline {
    private static final String SHADER_DIR = "/caustica/rt/";
    /** Set 1: entity albedo plus three independently indexed canonical material-page arrays. */
    private static final int BINDLESS_BINDINGS = 4;
    private static final int ENTITY_ALBEDO_BINDING = 0;
    private static final int MATERIAL_SURFACE0_BINDING = 1;
    private static final int MATERIAL_NORMAL_AO_BINDING = 2;
    private static final int MATERIAL_SURFACE1_BINDING = 3;
    // A ring of descriptor sets: setTlas writes the next slot (long-unused) rather than mutating the
    // slot in-flight frames are still reading, so the TLAS can be swapped without a device drain.
    // The TLAS is rebuilt + rebound every frame (dynamic content), so a slot is reused every RING
    // frames; RING must exceed the max frames-in-flight (vanilla MC ≤ 3) for the reused slot to be off
    // all queues. 6 gives margin and matches the KEEP_FRAMES-style horizon used for resource frees.
    private static final int RING = 6;

    private final RtContext ctx;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long[] descriptorSets;
    private int currentSet;
    private final long pipelineLayout;
    private final long pipeline;
    private final RtBuffer sbt;
    private final long sbtStride;
    private final int missCount;
    private final int hitGroupCount;
    private final int pushConstantSize;
    private final int pushConstantStages;
    // Optional second descriptor set (set 1) holding entity albedo and canonical material-page arrays.
    // Only entity albedo is update-after-bind: its RenderType→slot registry is append-only. Material
    // pages are populated once at the resource-epoch boundary. 0 when created without bindless textures.
    private final long bindlessLayout;
    private final long bindlessPool;
    private final long bindlessSet;
    private boolean destroyed;

    private RtPipeline(RtContext ctx, long dsl, long pool, long[] sets, long layout, long pipeline, RtBuffer sbt, long stride, int missCount, int hitGroupCount, int pushConstantSize, int pushConstantStages,
                       long bindlessLayout, long bindlessPool, long bindlessSet) {
        this.ctx = ctx;
        this.descriptorSetLayout = dsl;
        this.descriptorPool = pool;
        this.descriptorSets = sets;
        this.currentSet = 0;
        this.pipelineLayout = layout;
        this.pipeline = pipeline;
        this.sbt = sbt;
        this.sbtStride = stride;
        this.missCount = missCount;
        this.hitGroupCount = hitGroupCount;
        this.pushConstantSize = pushConstantSize;
        this.pushConstantStages = pushConstantStages;
        this.bindlessLayout = bindlessLayout;
        this.bindlessPool = bindlessPool;
        this.bindlessSet = bindlessSet;
    }

    /**
     * Builds the RT pipeline. {@code rahit} (nullable) adds any-hit-capable triangle hit records. With the
     * world pipeline, the hit SBT region is laid out to match {@link RtAccel}'s terrain bucket/ray-type
     * constants: radiance records first, shadow records second, then entity records. {@code extraStorageImages}
     * adds that many raygen-visible storage images at bindings 3.. (the DLSS-RR guide buffers);
     * write them with {@link #setExtraStorageImage}.
     */
    public static RtPipeline create(RtContext ctx, String rgen, String[] rmiss, String rchit, String rahit, int pushConstantSize, int bindlessTextures) {
        VkDevice vk = ctx.vk();
        boolean hasAhit = rahit != null;
        String label = "world RT pipeline";
        if (bindlessTextures > 0) {
            long requiredCombinedSamplers = Math.addExact(
                    Math.multiplyExact((long) bindlessTextures, BINDLESS_BINDINGS),
                    1L);
            long deviceLimit = ctx.updateAfterBindCombinedImageSamplerLimit();
            if (requiredCombinedSamplers > deviceLimit) {
                throw new UnsupportedOperationException("Configured bindless texture capacity " + bindlessTextures
                        + " requires " + requiredCombinedSamplers + " combined image samplers, device limit is "
                        + deviceLimit);
            }
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {

            LongBuffer p = stack.mallocLong(1); //在内存上临时创建1个long，用于后续从VK10里薅信息

            LongBuffer layouts = stack.mallocLong(RING); //获取布局句柄！
                {
                    VkDescriptorSetLayoutBinding.Buffer binds = VkDescriptorSetLayoutBinding.calloc(12, stack); //索引0-11，共12个绑定
                        binds.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                        binds.get(1).binding(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                        binds.get(2).binding(2).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | (hasAhit ? VK_SHADER_STAGE_ANY_HIT_BIT_KHR : 0));
                        binds.get(3).binding(3).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                        binds.get(4).binding(4).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                        binds.get(5).binding(5).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                        binds.get(6).binding(6).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                        binds.get(7).binding(7).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                        binds.get(8).binding(8).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                        binds.get(9).binding(9).descriptorType(VK10.VK_DESCRIPTOR_TYPE_SAMPLER) //3D采样器
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                        binds.get(10).binding(10).descriptorType(VK10.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)     //云噪声纹理
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                        binds.get(11).binding(11).descriptorType(VK10.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)     //水面噪声纹理
                                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);

                    
                        { //往pLayout里存布局句柄！
                            VkDescriptorSetLayoutCreateInfo dslci = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(binds); //创建布局句柄
                            check(VK10.vkCreateDescriptorSetLayout(vk, dslci, null, p), "vkCreateDescriptorSetLayout"); //写入p
                        }
                    
                    //往layouts里存全部的布局句柄！
                    long dsl = p.get(0); //拿到pLayout里的布局句柄
                    for (int i = 0; i < RING; i++) { //为RING个描述符集全部分配布局句柄
                        layouts.put(i, dsl);
                    }
                }


            long pool; //获取池句柄！
                {
                    VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(5, stack); //总共有5个
                        poolSizes.get(0).type(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(RING);  //加速结构
                        poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(RING * 7);      //输出图像，outImage到gSpecMotion共7个，在rgen里
                        poolSizes.get(2).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(RING); //方块图集
                        poolSizes.get(3).type(VK10.VK_DESCRIPTOR_TYPE_SAMPLER).descriptorCount(RING);                //噪声采样器
                        poolSizes.get(4).type(VK10.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).descriptorCount(RING * 2);          //噪声纹理

                    LongBuffer pPool = stack.mallocLong(1); //在内存上临时创建1个long，用于存句柄
                        { //往pPool里存池句柄！
                            VkDescriptorPoolCreateInfo dpci = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(RING).pPoolSizes(poolSizes); //创建池句柄
                            check(VK10.vkCreateDescriptorPool(vk, dpci, null, pPool), "vkCreateDescriptorPool"); //写入pPool
                        }
                    
                    //往pool里存全部的池句柄！
                    pool = pPool.get(0); //拿到pPool里的池句柄！所有的描述符集都从这个池里分配，不需要RING个池
                }


            long[] sets = new long[RING]; //获取描述符集！ 描述符集依赖布局句柄和池句柄，所以最后获取
                {
                    LongBuffer pSets = stack.mallocLong(RING); //长度不一样，不能用原来的p薅信息了。pSets为了承接VK10的输出，必须在原生内存上创建。sets是java堆上的数组，由Java类（this.descriptorSets）决定该变量是否存活，方便后续使用
                    VkDescriptorSetAllocateInfo dsai = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default().descriptorPool(pool).pSetLayouts(layouts);
                    check(VK10.vkAllocateDescriptorSets(vk, dsai, pSets), "vkAllocateDescriptorSets"); //写入pSets
                    pSets.get(sets); //sets[i] = pSets.get(i); 拷贝
                }


            long bindlessLayout = 0L, bindlessPool = 0L, bindlessSet = 0L;
            if (bindlessTextures > 0) {
                int nb = BINDLESS_BINDINGS;
                VkDescriptorSetLayoutBinding.Buffer bl = VkDescriptorSetLayoutBinding.calloc(nb, stack);
                java.nio.IntBuffer bindFlags = stack.mallocInt(nb);
                for (int i = 0; i < nb; i++) {
                    int stages = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
                    if (i == ENTITY_ALBEDO_BINDING && hasAhit) stages |= VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
                    bl.get(i).binding(i).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(bindlessTextures).stageFlags(stages);
                    int flags = VK12.VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
                    if (i == ENTITY_ALBEDO_BINDING) flags |= VK12.VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
                    bindFlags.put(i, flags);
                }
                VkDescriptorSetLayoutBindingFlagsCreateInfo bf = VkDescriptorSetLayoutBindingFlagsCreateInfo.calloc(stack).sType$Default().pBindingFlags(bindFlags);
                VkDescriptorSetLayoutCreateInfo bdslci = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pNext(bf.address()).flags(VK12.VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT).pBindings(bl);
                check(VK10.vkCreateDescriptorSetLayout(vk, bdslci, null, p), "vkCreateDescriptorSetLayout(bindless)");
                bindlessLayout = p.get(0);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, bindlessLayout, label + " bindless descriptor set layout");

                VkDescriptorPoolSize.Buffer bps = VkDescriptorPoolSize.calloc(1, stack);
                bps.get(0).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(bindlessTextures * nb);
                VkDescriptorPoolCreateInfo bdpci = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().flags(VK12.VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT).maxSets(1).pPoolSizes(bps);
                check(VK10.vkCreateDescriptorPool(vk, bdpci, null, p), "vkCreateDescriptorPool(bindless)");
                bindlessPool = p.get(0);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, bindlessPool, label + " bindless descriptor pool");

                VkDescriptorSetAllocateInfo bdsai = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default().descriptorPool(bindlessPool).pSetLayouts(stack.longs(bindlessLayout));
                check(VK10.vkAllocateDescriptorSets(vk, bdsai, p), "vkAllocateDescriptorSets(bindless)");
                bindlessSet = p.get(0);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, bindlessSet, label + " bindless descriptor set");
            }

            VkPipelineLayoutCreateInfo plci = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(bindlessTextures > 0 ? stack.longs(layouts.get(0), bindlessLayout) : stack.longs(layouts.get(0))); //layouts.get(0)就是dsl
            // Push constants are visible to raygen + closest-hit + miss (+ any-hit when present).
            // vkCmdPushConstants must be called with exactly these stages, so store them for trace().
            // Miss reads pc for the dynamic sky; widening the stage mask is the whole cost — no gotcha #3.
            int pcStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                    | VK_SHADER_STAGE_MISS_BIT_KHR
                    | (hasAhit ? VK_SHADER_STAGE_ANY_HIT_BIT_KHR : 0);
            if (pushConstantSize > 0) {
                VkPushConstantRange.Buffer pcr = VkPushConstantRange.calloc(1, stack)
                        .stageFlags(pcStages)
                        .offset(0).size(pushConstantSize);
                plci.pPushConstantRanges(pcr);
            }
            check(VK10.vkCreatePipelineLayout(vk, plci, null, p), "vkCreatePipelineLayout");
            long layout = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, label + " pipeline layout");

            // Stages: raygen, one miss per rmiss entry, the closest-hit, then (optionally) the any-hit.
            // Groups are raygen + N miss + the hit records selected by traceRayEXT's SBT offset/stride.
            int missCount = rmiss.length;
            int hitGroupCount = hasAhit ? RtAccel.SBT_HIT_GROUP_COUNT : 1;
            int groupCount = 1 + missCount + hitGroupCount;
            int hitGroupIdx = 1 + missCount;
            int chitStage = 1 + missCount;
            int ahitStage = chitStage + 1;
            int stageCount = 1 + missCount + 1 + (hasAhit ? 1 : 0);
            long mGen = loadModule(vk, stack, rgen);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, mGen, label + " " + rgen);
            long[] mMiss = new long[missCount];
            for (int m = 0; m < missCount; m++) {
                mMiss[m] = loadModule(vk, stack, rmiss[m]);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, mMiss[m], label + " " + rmiss[m]);
            }
            long mHit = loadModule(vk, stack, rchit);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, mHit, label + " " + rchit);
            long mAhit = hasAhit ? loadModule(vk, stack, rahit) : 0L;
            if (hasAhit) {
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, mAhit, label + " " + rahit);
            }
            ByteBuffer entry = stack.UTF8("main");
            VkPipelineShaderStageCreateInfo.Buffer stages = VkPipelineShaderStageCreateInfo.calloc(stageCount, stack);
            stages.get(0).sType$Default().stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR).module(mGen).pName(entry);
            for (int m = 0; m < missCount; m++) {
                stages.get(1 + m).sType$Default().stage(VK_SHADER_STAGE_MISS_BIT_KHR).module(mMiss[m]).pName(entry);
            }
            stages.get(chitStage).sType$Default().stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR).module(mHit).pName(entry);
            if (hasAhit) {
                stages.get(ahitStage).sType$Default().stage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR).module(mAhit).pName(entry);
            }

            VkRayTracingShaderGroupCreateInfoKHR.Buffer groups = VkRayTracingShaderGroupCreateInfoKHR.calloc(groupCount, stack);
            groups.get(0).sType$Default().type(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR)
                    .generalShader(0).closestHitShader(VK_SHADER_UNUSED_KHR).anyHitShader(VK_SHADER_UNUSED_KHR).intersectionShader(VK_SHADER_UNUSED_KHR);
            for (int m = 0; m < missCount; m++) {
                groups.get(1 + m).sType$Default().type(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR)
                        .generalShader(1 + m).closestHitShader(VK_SHADER_UNUSED_KHR).anyHitShader(VK_SHADER_UNUSED_KHR).intersectionShader(VK_SHADER_UNUSED_KHR);
            }
            for (int h = 0; h < hitGroupCount; h++) {
                groups.get(hitGroupIdx + h).sType$Default().type(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR)
                        .generalShader(VK_SHADER_UNUSED_KHR).closestHitShader(chitStage)
                        .anyHitShader(hasAhit && hitGroupUsesAnyHit(h) ? ahitStage : VK_SHADER_UNUSED_KHR)
                        .intersectionShader(VK_SHADER_UNUSED_KHR);
            }

            VkRayTracingPipelineCreateInfoKHR.Buffer rtpci = VkRayTracingPipelineCreateInfoKHR.calloc(1, stack);
            // Depth 1: secondary shadow/visibility rays are issued sequentially from raygen (not
            // nested in closest-hit), so each traceRayEXT is depth 1 — no recursion budget needed.
            rtpci.get(0).sType$Default().pStages(stages).pGroups(groups).maxPipelineRayRecursionDepth(1).layout(layout);
            if (RtDeviceBringup.ommEnabled()) {
                rtpci.get(0).flags(VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT);
            }
            LongBuffer pPipeline = stack.mallocLong(1);
            check(vkCreateRayTracingPipelinesKHR(vk, VK10.VK_NULL_HANDLE, VK10.VK_NULL_HANDLE, rtpci, null, pPipeline),
                    "vkCreateRayTracingPipelinesKHR");
            long pipeline = pPipeline.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pipeline, label);

            VK10.vkDestroyShaderModule(vk, mGen, null);
            for (int m = 0; m < missCount; m++) {
                VK10.vkDestroyShaderModule(vk, mMiss[m], null);
            }
            VK10.vkDestroyShaderModule(vk, mHit, null);
            if (hasAhit) {
                VK10.vkDestroyShaderModule(vk, mAhit, null);
            }

            // SBT: one record per group. Over-align the stride so every region start is base-aligned and
            // every individual record satisfies shaderGroupHandleAlignment.
            int handleSize = ctx.shaderGroupHandleSize();
            ByteBuffer handles = stack.malloc(groupCount * handleSize);
            check(vkGetRayTracingShaderGroupHandlesKHR(vk, pipeline, 0, groupCount, handles), "vkGetRayTracingShaderGroupHandlesKHR");
            long stride = align(handleSize,
                    Math.max(ctx.shaderGroupBaseAlignment(), ctx.shaderGroupHandleAlignment()));
            if (stride > Integer.toUnsignedLong(ctx.maxShaderGroupStride())) {
                throw new UnsupportedOperationException("SBT stride " + stride + " exceeds maxShaderGroupStride "
                        + Integer.toUnsignedLong(ctx.maxShaderGroupStride()));
            }
            RtBuffer sbt = ctx.createAlignedBuffer(stride * groupCount,
                    VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR, true,
                    label + " shader binding table", ctx.shaderGroupBaseAlignment());
            for (int g = 0; g < groupCount; g++) {
                MemoryUtil.memCopy(MemoryUtil.memAddress(handles) + (long) g * handleSize, sbt.mapped + g * stride, handleSize);
            }
            sbt.flush();
            return new RtPipeline(ctx, layouts.get(0), pool, sets, layout, pipeline, sbt, stride, missCount, hitGroupCount, pushConstantSize, pcStages,
                    bindlessLayout, bindlessPool, bindlessSet);
        }
    }

    private static boolean hitGroupUsesAnyHit(int relativeHitGroup) {
        if (relativeHitGroup < RtAccel.SBT_ENTITY_OFFSET) {
            int rayType = relativeHitGroup / RtAccel.TERRAIN_BUCKETS;
            int bucket = relativeHitGroup % RtAccel.TERRAIN_BUCKETS;
            if (rayType == RtAccel.SBT_RAY_RADIANCE) {
                return bucket == RtAccel.BUCKET_CUTOUT;
            }
            return bucket != RtAccel.BUCKET_SOLID;
        }
        int entityBucket = (relativeHitGroup - RtAccel.SBT_ENTITY_OFFSET) % RtAccel.TERRAIN_BUCKETS;
        return entityBucket == RtAccel.ENTITY_BUCKET_ANY_HIT;
    }

    /**
     * Bind a new TLAS into the next ring slot (which in-flight frames are no longer reading, since
     * swaps are many frames apart) and make it current, so the binding can change without a drain.
     */
    public void setTlas(long tlas) {
        currentSet = (currentSet + 1) % RING;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkWriteDescriptorSetAccelerationStructureKHR asWrite = VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR).pAccelerationStructures(stack.longs(tlas));
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack);
            write.get(0).sType$Default().pNext(asWrite.address()).dstSet(descriptorSets[currentSet]).dstBinding(0)
                    .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
            VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
        }
    }

    /** Write the storage image into every ring slot (set once at init / on resize, when idle). */
    public void setStorageImage(long imageView) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer imgInfo = VkDescriptorImageInfo.calloc(1, stack);
            imgInfo.get(0).imageView(imageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(RING, stack);
            for (int i = 0; i < RING; i++) {
                write.get(i).sType$Default().dstSet(descriptorSets[i]).dstBinding(1)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(imgInfo);
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
        }
    }

    /** Bind the block albedo atlas into every ring slot. */
    public void setBlockAlbedoAtlas(long sampler, long imageView) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer info = VkDescriptorImageInfo.calloc(1, stack);
            info.get(0).sampler(sampler).imageView(imageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(RING, stack);
            for (int i = 0; i < RING; i++) {
                write.get(i).sType$Default()
                        .dstSet(descriptorSets[i])
                        .dstBinding(2)
                        .descriptorCount(1)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(info);
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
        }
    }

    /** Write an extra storage image (DLSS-RR guide buffer) into binding {@code firstExtraBinding + slot} across every ring slot. */
    public void setExtraStorageImage(long imageView, int binding) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer info = VkDescriptorImageInfo.calloc(1, stack);
            info.get(0).imageView(imageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(RING, stack);
            for (int i = 0; i < RING; i++) {
                write.get(i).sType$Default()
                        .dstSet(descriptorSets[i])
                        .dstBinding(binding) //绑定到指定编号
                        .descriptorCount(1)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(info);
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
        }
    }

    /** 绑定 3D 采样器到所有 RING 个描述符集。 */
    public void setSampler3D(long sampler, int binding) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer info = VkDescriptorImageInfo.calloc(1, stack);
            info.get(0).sampler(sampler);
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(RING, stack);
            for (int i = 0; i < RING; i++) {
                write.get(i).sType$Default()
                        .dstSet(descriptorSets[i])
                        .dstBinding(binding) //绑定到指定编号
                        .dstArrayElement(0)
                        .descriptorCount(1)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_SAMPLER) //对应slang侧的 SamplerState
                        .pImageInfo(info);
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
        }
    }

    /** 绑定纹理到所有 RING 个描述符集。 */
    public void setTexture3D(long imageView, int binding) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer info = VkDescriptorImageInfo.calloc(1, stack);
            info.get(0).imageView(imageView).imageLayout(VK10.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(RING, stack);
            for (int i = 0; i < RING; i++) {
                write.get(i).sType$Default()
                        .dstSet(descriptorSets[i])
                        .dstBinding(binding) //绑定到指定编号
                        .dstArrayElement(0)
                        .descriptorCount(1)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
                        .pImageInfo(info);
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
        }
    }



    /** Append or initialize one entity-albedo slot. Existing slots never change while frames are in flight. */
    public void setEntityAlbedoTexture(int slot, long imageView, long sampler) {
        setBindlessTexture(ENTITY_ALBEDO_BINDING, slot, imageView, sampler);
    }

    /** Bind one compact canonical page bundle at a resource-epoch boundary. */
    public void setMaterialPage(int page, long surface0View, long normalAoView, long surface1View,
                                long sampler) {
        setBindlessTexture(MATERIAL_SURFACE0_BINDING, page, surface0View, sampler);
        setBindlessTexture(MATERIAL_NORMAL_AO_BINDING, page, normalAoView, sampler);
        setBindlessTexture(MATERIAL_SURFACE1_BINDING, page, surface1View, sampler);
    }

    private void setBindlessTexture(int binding, int slot, long imageView, long sampler) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer info = VkDescriptorImageInfo.calloc(1, stack);
            info.get(0).sampler(sampler).imageView(imageView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack);
            write.get(0).sType$Default().dstSet(bindlessSet).dstBinding(binding).dstArrayElement(slot)
                    .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(info);
            VK10.vkUpdateDescriptorSets(ctx.vk(), write, null);
        }
    }

    /** True if this pipeline was created with a bindless entity-texture set. */
    public boolean hasBindless() {
        return bindlessSet != 0L;
    }

    public void trace(VkCommandBuffer cmd, int width, int height) {
        trace(cmd, width, height, null);
    }

    /** Record bind (+ optional raygen push constants) + trace into the given command buffer. */
    public void trace(VkCommandBuffer cmd, int width, int height, java.nio.ByteBuffer pushConstants) {
        try (MemoryStack stack = MemoryStack.stackPush(); RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "trace rays")) {
            VK10.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
            java.nio.LongBuffer boundSets = bindlessSet != 0L
                    ? stack.longs(descriptorSets[currentSet], bindlessSet)
                    : stack.longs(descriptorSets[currentSet]);
            VK10.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0, boundSets, null);
            if (pushConstants != null && pushConstantSize > 0) {
                VK10.vkCmdPushConstants(cmd, pipelineLayout, pushConstantStages, 0, pushConstants);
            }
            VkStridedDeviceAddressRegionKHR raygen = VkStridedDeviceAddressRegionKHR.calloc(stack)
                    .deviceAddress(sbt.deviceAddress).stride(sbtStride).size(sbtStride);
            VkStridedDeviceAddressRegionKHR miss = VkStridedDeviceAddressRegionKHR.calloc(stack)
                    .deviceAddress(sbt.deviceAddress + sbtStride).stride(sbtStride).size((long) missCount * sbtStride);
            VkStridedDeviceAddressRegionKHR hit = VkStridedDeviceAddressRegionKHR.calloc(stack)
                    .deviceAddress(sbt.deviceAddress + (1L + missCount) * sbtStride).stride(sbtStride).size((long) hitGroupCount * sbtStride);
            VkStridedDeviceAddressRegionKHR callable = VkStridedDeviceAddressRegionKHR.calloc(stack);
            vkCmdTraceRaysKHR(cmd, raygen, miss, hit, callable, width, height, 1);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        sbt.destroy();
        VK10.vkDestroyPipeline(vk, pipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        if (bindlessPool != 0L) {
            VK10.vkDestroyDescriptorPool(vk, bindlessPool, null);
        }
        if (bindlessLayout != 0L) {
            VK10.vkDestroyDescriptorSetLayout(vk, bindlessLayout, null);
        }
        destroyed = true;
    }

    private static long align(long v, long a) {
        return (v + a - 1) & ~(a - 1);
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtPipeline.class.getResourceAsStream(SHADER_DIR + name)) {
            if (in == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER_DIR + name);
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + SHADER_DIR + name, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo smci = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
            LongBuffer pModule = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, smci, null, pModule), "vkCreateShaderModule(" + name + ")");
            return pModule.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
