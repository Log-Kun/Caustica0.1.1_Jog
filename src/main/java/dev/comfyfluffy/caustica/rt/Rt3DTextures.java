package dev.comfyfluffy.caustica.rt;

import com.mojang.blaze3d.platform.NativeImage;

import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.terrain.RtTerrain;
import net.minecraft.client.Minecraft;
import net.minecraft.core.BlockPos;
import net.minecraft.resources.Identifier;
import net.minecraft.server.packs.resources.Resource;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.vma.Vma;
import org.lwjgl.util.vma.VmaAllocationCreateInfo;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkBufferImageCopy;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkImageCreateInfo;
import org.lwjgl.vulkan.VkImageMemoryBarrier;
import org.lwjgl.vulkan.VkImageViewCreateInfo;
import org.lwjgl.vulkan.VkSamplerCreateInfo;

public class Rt3DTextures {

    private static NativeImage loadPng(String path) {
        try {
            Identifier id = Identifier.fromNamespaceAndPath("caustica", path); //读src\main\resources\assets\caustica\textures内
            Resource resource = Minecraft.getInstance().getResourceManager().getResourceOrThrow(id);
            try (InputStream in = resource.open()) {
                return NativeImage.read(in);
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load noise texture: " + path, e);
        }
    }


    public static RtImage createCloudNoiseTexture(RtContext ctx) {
        
        NativeImage img = loadPng("textures/cloud_noise_3d_128.png");

        int size = img.getHeight();
        int slices = img.getWidth() / size;
        int width = img.getWidth();
        int totalBytes = size * size * slices;
        
        RtBuffer staging = ctx.createBuffer(
            (long) totalBytes,        // 128³ 灰度图 = 2 MB
            VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            true,
            "noise staging");

        int[] pixels = img.getPixelsABGR(); //像素点横向展开一排排，xyz分别代表横竖深

        int i = 0;
        for (int z = 0; z < slices; z++) { //z是深度方向
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    MemoryUtil.memPutByte(staging.mapped + i, (byte)(pixels[y * width + z * size + x] & 0xFF)); //获取像素颜色，只取R通道，存入 staging buffer，1byte。mapped是CPU可写的地址
                    i ++;
                }
            }
        }
        
        staging.flush(0, totalBytes);
        
        RtImage noiseImage = ctx.createTexture3D(size, size, slices, VK10.VK_FORMAT_R8_UNORM, "noise"); //创建3D纹理，1byte/像素，灰度图， layout 暂时为 UNDEFINED
        ctx.uploadTo3DImage(size, size, slices, staging, noiseImage.image); // staging 拷贝到 noiseImage.image （GPU侧的3D纹理），并把纹理的 layout 从 UNDEFINED 转为 SHADER_READ_ONLY_OPTIMAL
        
        staging.destroy(); //销毁 staging，数据传入gpu后就不需要了
        return noiseImage;
    }



    public static RtImage createWaterNoiseTexture(RtContext ctx) {
        
        NativeImage img = loadPng("textures/water_wave_3d_128.png");

        int size = img.getHeight();
        int slices = img.getWidth() / size;
        int width = img.getWidth();
        int totalBytes = size * size * slices * 2; // RG8 两个通道
        
        RtBuffer staging = ctx.createBuffer(
            (long) totalBytes,        // 128³ RG8图 = 4 MB
            VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            true,
            "noise staging");

        int[] pixels = img.getPixelsABGR(); //像素点横向展开一排排，xyz分别代表横竖深

        int i = 0;
        for (int z = 0; z < slices; z++) { //z是深度方向
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    int pixel = pixels[y * width + z * size + x];
                    MemoryUtil.memPutByte(staging.mapped + i,     (byte)(pixel & 0xFF));         // R
                    MemoryUtil.memPutByte(staging.mapped + i + 1, (byte)((pixel >> 8) & 0xFF));  // G
                    i += 2;
                }
            }
        }
        
        staging.flush(0, totalBytes);
        
        RtImage noiseImage = ctx.createTexture3D(size, size, slices, VK10.VK_FORMAT_R8G8_UNORM, "noise"); //创建3D纹理，1byte/像素，灰度图， layout 暂时为 UNDEFINED
        ctx.uploadTo3DImage(size, size, slices, staging, noiseImage.image); // staging 拷贝到 noiseImage.image （GPU侧的3D纹理），并把纹理的 layout 从 UNDEFINED 转为 SHADER_READ_ONLY_OPTIMAL
        
        staging.destroy(); //销毁 staging，数据传入gpu后就不需要了
        return noiseImage;
    }





    public static long create3DSampler(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSamplerCreateInfo sci = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR)
                    .minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT); //多一个w轴寻址配置
            LongBuffer p = stack.mallocLong(1);
            if (VK10.vkCreateSampler(ctx.vk(), sci, null, p) != VK10.VK_SUCCESS) { //p里存放创建的 sampler 的 handle，若失败则抛报错
                throw new IllegalStateException("vkCreateSampler failed");
            }
            return p.get(0);
        }
    }
}