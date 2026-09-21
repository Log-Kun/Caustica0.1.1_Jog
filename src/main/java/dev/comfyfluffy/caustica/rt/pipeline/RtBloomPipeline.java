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
    private final RtContext ctx;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long pipelineLayout;
    private final long bloom0DescriptorSet;
    private final long bloom1DescriptorSet;
    private final long bloom2DescriptorSet;
    private final long bloom3DescriptorSet;
    private final long bloom4DescriptorSet;
    private final long bloom5DescriptorSet;
    private final long bloom6DescriptorSet;
    private final long bloom0Pipeline;
    private final long bloom1Pipeline;
    private final long bloom2Pipeline;
    private final long bloom3Pipeline;
    private final long bloom4Pipeline;
    private final long bloom5Pipeline;
    private final long bloom6Pipeline;

    private boolean destroyed;

    private RtBloomPipeline( ////RtBloomPipeline是个普通JAVA类，可以理解为巨型的自定义数据类型，这里是它的定义
            RtContext ctx,
            long dsl,
            long pool,
            long layout,
            long bloom0DescriptorSet,
            long bloom1DescriptorSet,
            long bloom2DescriptorSet,
            long bloom3DescriptorSet,
            long bloom4DescriptorSet,
            long bloom5DescriptorSet,
            long bloom6DescriptorSet,
            long bloom0Pipeline,
            long bloom1Pipeline,
            long bloom2Pipeline,
            long bloom3Pipeline,
            long bloom4Pipeline,
            long bloom5Pipeline,
            long bloom6Pipeline) {
        this.ctx = ctx;
        descriptorSetLayout = dsl;
        descriptorPool = pool;
        pipelineLayout = layout;
        this.bloom0DescriptorSet = bloom0DescriptorSet;
        this.bloom1DescriptorSet = bloom1DescriptorSet;
        this.bloom2DescriptorSet = bloom2DescriptorSet;
        this.bloom3DescriptorSet = bloom3DescriptorSet;
        this.bloom4DescriptorSet = bloom4DescriptorSet;
        this.bloom5DescriptorSet = bloom5DescriptorSet;
        this.bloom6DescriptorSet = bloom6DescriptorSet;
        this.bloom0Pipeline = bloom0Pipeline;
        this.bloom1Pipeline = bloom1Pipeline;
        this.bloom2Pipeline = bloom2Pipeline;
        this.bloom3Pipeline = bloom3Pipeline;
        this.bloom4Pipeline = bloom4Pipeline;
        this.bloom5Pipeline = bloom5Pipeline;
        this.bloom6Pipeline = bloom6Pipeline;
    }

    public static RtBloomPipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk(); //返回 Vulkan 逻辑设备句柄，说明要跑在哪个gpu上
        try (MemoryStack stack = MemoryStack.stackPush()) { //创建stack对象（堆在原生内存上），存放要传给vk的结构体。.stackPush()对准当前最近的一块可用内存（紧贴着已经使用的内存）。这部分内存try之后直接释放
            VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(2, stack); //调用 VkDescriptorSetLayoutBinding 的静态方法 calloc，在栈上分配够存 2 个VkDescriptorSetLayoutBinding结构体的空间，返回一个数组缓冲区，用 VkDescriptorSetLayoutBinding.Buffer 类型的变量 bindings 来接收
            
            //在刚刚分配的栈上创建两个绑定图像，一个用于输出图像，一个用于输入图像,这里仅声明有这俩东西
            bindings.get(0).binding(0).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            bindings.get(1).binding(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack) //分配一段大小适合VkDescriptorSetLayoutCreateInfo类型的内存。这里不是.Buffer，所以calloc没有参数（默认为1）
                    .sType$Default().pBindings(bindings); //.sType$Default可以自动填入正确的结构体类型常量，pBindings是个指针，指向bindings
            
            LongBuffer handle = stack.mallocLong(1); //在内存上创建一个长度为1的long数组，后面用来存创建结果。handle指向一块原生内存，后面.get()时会自动解引用
            check(VK10.vkCreateDescriptorSetLayout(vk, layoutInfo, null, handle), //此函数最后一个传入需要是内存地址，handle作为输出参数，必须提前分配好一块内存让函数写
                    "vkCreateDescriptorSetLayout(rt bloom)"); //check()用于检查是否出错，如果出错，返回"vkCreateDescriptorSetLayout(rt bloom)"这样的信息
            long dsl = handle.get(0); //取出信息 这里存的是描述符集布局句柄 都是些抽象玩意，记得类型是long就行了
            //上面四行是个很经典的函数写回结果的方法

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(1, stack);
                poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(14); //填充poolSizes.get(0)，内存池中预留14个STORAGE_IMAGE的空间 因为泛光部分0123456七个pass，每个2图像（输入输出），7*2=14
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(7).pPoolSizes(poolSizes); //最多可分配7个描述符集，每个pass前所有的layout(binding = 0, set = 0, rgba16f)们是一个描述符集。pPoolSizes(poolSizes)是指向poolSizes地址的指针，池容量从这里读
            VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle); //类似地，复用handle取出信息
            long pool = handle.get(0);

            VkDescriptorSetAllocateInfo allocInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(pool).pSetLayouts(stack.longs(dsl, dsl, dsl, dsl, dsl, dsl, dsl)); //按照描述符集布局句柄创建 创建几个下面就用几个
            LongBuffer setHandles = stack.mallocLong(7);
            VK10.vkAllocateDescriptorSets(vk, allocInfo, setHandles); //用setHandles存一堆信息，后面用.get()拆开读取
            long bloom0DescriptorSet = setHandles.get(0);
            long bloom1DescriptorSet = setHandles.get(1);
            long bloom2DescriptorSet = setHandles.get(2);
            long bloom3DescriptorSet = setHandles.get(3);
            long bloom4DescriptorSet = setHandles.get(4);
            long bloom5DescriptorSet = setHandles.get(5);
            long bloom6DescriptorSet = setHandles.get(6);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(dsl));
            VK10.vkCreatePipelineLayout(vk, pipelineLayoutInfo, null, handle);
            long layout = handle.get(0);

                long bloom0Pipeline = createComputePipeline(vk, stack, layout, "bloom0.comp.spv");
                long bloom1Pipeline = createComputePipeline(vk, stack, layout, "bloom1.comp.spv");
                long bloom2Pipeline = createComputePipeline(vk, stack, layout, "bloom2.comp.spv");
                long bloom3Pipeline = createComputePipeline(vk, stack, layout, "bloom3.comp.spv");
                long bloom4Pipeline = createComputePipeline(vk, stack, layout, "bloom4.comp.spv");
                long bloom5Pipeline = createComputePipeline(vk, stack, layout, "bloom5.comp.spv");
                long bloom6Pipeline = createComputePipeline(vk, stack, layout, "bloom6.comp.spv");

                return new RtBloomPipeline(ctx, dsl, pool, layout,
                    bloom0DescriptorSet, bloom1DescriptorSet, bloom2DescriptorSet,
                    bloom3DescriptorSet, bloom4DescriptorSet, bloom5DescriptorSet, bloom6DescriptorSet,
                    bloom0Pipeline, bloom1Pipeline, bloom2Pipeline,
                    bloom3Pipeline, bloom4Pipeline, bloom5Pipeline, bloom6Pipeline); //返回一个 RtBloomPipeline 实例，包含管线信息
                    //RtBloomPipeline是个普通JAVA类，可以理解为巨型的自定义数据类型，其定义在上面有
        }
    }

    public void setImages(long inputView, long bloom0View, long bloom1View, long bloom2View, long bloom3View, long bloom4View, long bloom5View, long bloom6View) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            writeSet(stack, bloom0DescriptorSet, bloom0View, inputView);
            writeSet(stack, bloom1DescriptorSet, bloom1View, bloom0View);
            writeSet(stack, bloom2DescriptorSet, bloom2View, bloom1View);
            writeSet(stack, bloom3DescriptorSet, bloom3View, bloom2View);
            writeSet(stack, bloom4DescriptorSet, bloom4View, bloom3View);
            writeSet(stack, bloom5DescriptorSet, bloom5View, bloom4View);
            writeSet(stack, bloom6DescriptorSet, bloom6View, bloom5View);
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

        public void dispatch(
            VkCommandBuffer cmd,
            int fullWidth,
            int fullHeight,
            int halfWidth,
            int halfHeight,
            int quarterWidth,
            int quarterHeight) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "bloom")) {
            VkMemoryBarrier.Buffer barrier = VkMemoryBarrier.calloc(1, stack).sType$Default()
                    .srcAccessMask(VK10.VK_ACCESS_SHADER_WRITE_BIT)
                    .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT | VK10.VK_ACCESS_SHADER_WRITE_BIT);
            bindAndDispatch(cmd, stack, bloom0Pipeline, bloom0DescriptorSet, fullWidth, fullHeight);
            insertBarrier(cmd, stack, barrier);
            bindAndDispatch(cmd, stack, bloom1Pipeline, bloom1DescriptorSet, fullWidth, fullHeight);
            insertBarrier(cmd, stack, barrier);
            bindAndDispatch(cmd, stack, bloom2Pipeline, bloom2DescriptorSet, fullWidth, fullHeight);
            insertBarrier(cmd, stack, barrier);
            bindAndDispatch(cmd, stack, bloom3Pipeline, bloom3DescriptorSet, halfWidth, halfHeight);
            insertBarrier(cmd, stack, barrier);
            bindAndDispatch(cmd, stack, bloom4Pipeline, bloom4DescriptorSet, halfWidth, halfHeight);
            insertBarrier(cmd, stack, barrier);
            bindAndDispatch(cmd, stack, bloom5Pipeline, bloom5DescriptorSet, quarterWidth, quarterHeight);
            insertBarrier(cmd, stack, barrier);
            bindAndDispatch(cmd, stack, bloom6Pipeline, bloom6DescriptorSet, quarterWidth, quarterHeight);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, bloom0Pipeline, null);
        VK10.vkDestroyPipeline(vk, bloom1Pipeline, null);
        VK10.vkDestroyPipeline(vk, bloom2Pipeline, null);
        VK10.vkDestroyPipeline(vk, bloom3Pipeline, null);
        VK10.vkDestroyPipeline(vk, bloom4Pipeline, null);
        VK10.vkDestroyPipeline(vk, bloom5Pipeline, null);
        VK10.vkDestroyPipeline(vk, bloom6Pipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        destroyed = true;
    }

    private void bindAndDispatch(
            VkCommandBuffer cmd,
            MemoryStack stack,
            long pipeline,
            long descriptorSet,
            int width,
            int height) {
        VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelineLayout, 0, stack.longs(descriptorSet), null);
        VK10.vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    }

    private static void insertBarrier(
            VkCommandBuffer cmd,
            MemoryStack stack,
            VkMemoryBarrier.Buffer barrier) {
        VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, barrier, null, null);
    }

    private static long createComputePipeline(
            VkDevice vk,
            MemoryStack stack,
            long pipelineLayout,
            String shaderName) {
        long module = loadModule(vk, stack, shaderName);
        try {
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                    .sType$Default()
                    .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(module)
                    .pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer pipelineInfo = VkComputePipelineCreateInfo.calloc(1, stack);
            pipelineInfo.get(0).sType$Default().stage(stage).layout(pipelineLayout);
            LongBuffer pipelineHandle = stack.mallocLong(1);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, pipelineInfo, null, pipelineHandle),
                    "vkCreateComputePipelines(rt bloom " + shaderName + ")");
            return pipelineHandle.get(0);
        } finally {
            VK10.vkDestroyShaderModule(vk, module, null);
        }
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