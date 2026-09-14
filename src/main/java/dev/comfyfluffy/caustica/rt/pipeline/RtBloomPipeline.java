package dev.comfyfluffy.caustica.rt.pipeline;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkMemoryBarrier;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.LongBuffer;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

/** Builds a quarter-resolution blurred bloom image. Final composition stays in bloomfinal.comp. */
public final class RtBloomPipeline {
    private static final String SHADER_DIR = "/caustica/rt/";
    private static final int PASS_COUNT = 6;

    private final RtContext ctx;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long[] descriptorSets;
    private final long pipelineLayout;
    private final long[] pipelines;
    private boolean destroyed;

    private RtBloomPipeline(RtContext ctx, long dsl, long pool, long[] sets, long layout, long[] pipelines) {
        this.ctx = ctx;
        descriptorSetLayout = dsl;
        descriptorPool = pool;
        descriptorSets = sets;
        pipelineLayout = layout;
        this.pipelines = pipelines;
    }

    public static RtBloomPipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(2, stack);
            for (int i = 0; i < 2; i++) {
                bindings.get(i).binding(i).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(bindings);
            LongBuffer handle = stack.mallocLong(1);
            check(VK10.vkCreateDescriptorSetLayout(vk, layoutInfo, null, handle),
                    "vkCreateDescriptorSetLayout(rt bloom)");
            long dsl = handle.get(0);

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(1, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(PASS_COUNT * 2);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(PASS_COUNT).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle),
                    "vkCreateDescriptorPool(rt bloom)");
            long pool = handle.get(0);

            VkDescriptorSetAllocateInfo allocInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(pool).pSetLayouts(stack.longs(dsl, dsl, dsl, dsl, dsl, dsl));
            LongBuffer setHandles = stack.mallocLong(PASS_COUNT);
            check(VK10.vkAllocateDescriptorSets(vk, allocInfo, setHandles),
                    "vkAllocateDescriptorSets(rt bloom)");
            long[] sets = new long[PASS_COUNT];
            for (int i = 0; i < PASS_COUNT; i++) {
                sets[i] = setHandles.get(i);
            }

            VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(dsl));
            check(VK10.vkCreatePipelineLayout(vk, pipelineLayoutInfo, null, handle),
                    "vkCreatePipelineLayout(rt bloom)");
            long layout = handle.get(0);

            String[] shaderNames = {
                    "bloom0.comp.spv",
                    "bloom1.comp.spv",
                    "bloom2.comp.spv",
                    "bloom3.comp.spv",
                    "bloom4.comp.spv",
                    "bloom5.comp.spv"
            };
            long[] pipelines = new long[PASS_COUNT];
            for (int i = 0; i < PASS_COUNT; i++) {
                long module = loadModule(vk, stack, shaderNames[i]);
                VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                        .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                        .module(module).pName(stack.UTF8("main"));
                VkComputePipelineCreateInfo.Buffer pipelineInfo = VkComputePipelineCreateInfo.calloc(1, stack);
                pipelineInfo.get(0).sType$Default().stage(stage).layout(layout);
                LongBuffer pipelineHandle = stack.mallocLong(1);
                check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, pipelineInfo, null, pipelineHandle),
                        "vkCreateComputePipelines(rt bloom " + i + ")");
                pipelines[i] = pipelineHandle.get(0);
                VK10.vkDestroyShaderModule(vk, module, null);
            }
            return new RtBloomPipeline(ctx, dsl, pool, sets, layout, pipelines);
        }
    }

    public void setImages(long inputView, long bloom0View, long bloom1View, long bloom2View, long bloom3View, long bloom4View, long bloom5View) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            writeSet(stack, descriptorSets[0], bloom0View, inputView);
            writeSet(stack, descriptorSets[1], bloom1View, bloom0View);
            writeSet(stack, descriptorSets[2], bloom2View, bloom1View);
            writeSet(stack, descriptorSets[3], bloom3View, bloom2View);
            writeSet(stack, descriptorSets[4], bloom4View, bloom3View);
            writeSet(stack, descriptorSets[5], bloom5View, bloom4View);
        }
    }

    private void writeSet(MemoryStack stack, long set, long output, long input) {
        VkDescriptorImageInfo.Buffer infos = VkDescriptorImageInfo.calloc(2, stack);
        infos.get(0).imageView(output).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        infos.get(1).imageView(input).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(2, stack);
        for (int i = 0; i < 2; i++) {
            writes.get(i).sType$Default().dstSet(set).dstBinding(i).descriptorCount(1)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(infos.slice(i, 1));
        }
        VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
    }

    public void dispatch(VkCommandBuffer cmd, int bloomWidth, int bloomHeight) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "bloom")) {
            int groupsX = (bloomWidth + 15) / 16;
            int groupsY = (bloomHeight + 15) / 16;
            VkMemoryBarrier.Buffer barrier = VkMemoryBarrier.calloc(1, stack).sType$Default()
                    .srcAccessMask(VK10.VK_ACCESS_SHADER_WRITE_BIT)
                    .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT | VK10.VK_ACCESS_SHADER_WRITE_BIT);
            for (int i = 0; i < PASS_COUNT; i++) {
                VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[i]);
                VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipelineLayout, 0, stack.longs(descriptorSets[i]), null);
                VK10.vkCmdDispatch(cmd, groupsX, groupsY, 1);
                if (i + 1 < PASS_COUNT) {
                    VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, barrier, null, null);
                }
            }
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        for (long pipeline : pipelines) {
            VK10.vkDestroyPipeline(vk, pipeline, null);
        }
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        destroyed = true;
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtBloomPipeline.class.getResourceAsStream(SHADER_DIR + name)) {
            if (in == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER_DIR + name);
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + SHADER_DIR + name, e);
        }
        var code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo info = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
            LongBuffer module = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, info, null, module), "vkCreateShaderModule(" + name + ")");
            return module.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}