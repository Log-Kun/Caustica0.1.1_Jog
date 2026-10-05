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

    public static long getSampler3D(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSamplerCreateInfo sci = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR)
                    .minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_REPEAT); //多一个w轴寻址配置
            LongBuffer p = stack.mallocLong(1);
            if (VK10.vkCreateSampler(ctx.vk(), sci, null, p) != VK10.VK_SUCCESS) { //用p薅出创建的 sampler 的 handle，若失败则抛报错
                throw new IllegalStateException("vkCreateSampler3D failed");
            }
            return p.get(0);
        }
    }




    private static NativeImage loadPng(String path) {
        try {
            Identifier id = Identifier.fromNamespaceAndPath("caustica", path); //读src\main\resources\assets\caustica\textures内
            Resource resource = Minecraft.getInstance().getResourceManager().getResourceOrThrow(id);
            try (InputStream in = resource.open()) {
                return NativeImage.read(in);
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load texture: " + path, e);
        }
    }


    public static long getTexture3D(RtContext ctx, String path, int format) {
        
        NativeImage img = loadPng(path);

        int size = img.getHeight();
        int slices = img.getWidth() / size;
        int width = img.getWidth();
        int channelNum = 1;

        if(format == VK10.VK_FORMAT_R8G8_UNORM) {
            channelNum = 2;
        }else if(format == VK10.VK_FORMAT_R8G8B8_UNORM) {
            channelNum = 3;
        }else if(format == VK10.VK_FORMAT_R8G8B8A8_UNORM) {
            channelNum = 4;
        }
        
        int totalBytes = size * size * slices * channelNum;

        RtBuffer staging = ctx.createBuffer((long) totalBytes, VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "noise staging");

        int[] pixels = img.getPixelsABGR(); //像素点横向展开一排排，xyz分别代表横竖深。ABGR，FF FF FF FF，8位 8位 8位 8位
        
        if(channelNum == 1){
            int i = 0;
            for (int z = 0; z < slices; z++) { //z是深度方向
                for (int y = 0; y < size; y++) {
                    for (int x = 0; x < size; x++) {
                        MemoryUtil.memPutByte(staging.mapped + i, (byte)(pixels[y * width + z * size + x] & 0x000000FF)); //R，获取像素颜色，存入 staging buffer，1byte。mapped是CPU可写的地址
                        i ++;
                    }
                }
            }
        }else if(channelNum == 2){
            int i = 0;
            for (int z = 0; z < slices; z++) {
                for (int y = 0; y < size; y++) {
                    for (int x = 0; x < size; x++) {
                        int pix = pixels[y * width + z * size + x];
                        MemoryUtil.memPutByte(staging.mapped + i, (byte)(pix & 0x000000FF));              //R
                        MemoryUtil.memPutByte(staging.mapped + i + 1, (byte)((pix >> 8) & 0x000000FF));   //G   >>8 意为把整个32位整数右移8位，&为与运算，提取位移后的最低位
                        i += 2;
                    }
                }
            }
        }else if(channelNum == 3){
            int i = 0;
            for (int z = 0; z < slices; z++) {
                for (int y = 0; y < size; y++) {
                    for (int x = 0; x < size; x++) {
                        int pix = pixels[y * width + z * size + x];
                        MemoryUtil.memPutByte(staging.mapped + i, (byte)(pix & 0x000000FF));              //R
                        MemoryUtil.memPutByte(staging.mapped + i + 1, (byte)((pix >> 8) & 0x000000FF));   //G
                        MemoryUtil.memPutByte(staging.mapped + i + 1, (byte)((pix >> 16) & 0x000000FF));  //B
                        i += 3;
                    }
                }
            }
        }else if(channelNum == 4){
            int i = 0;
            for (int z = 0; z < slices; z++) {
                for (int y = 0; y < size; y++) {
                    for (int x = 0; x < size; x++) {
                        int pix = pixels[y * width + z * size + x];
                        MemoryUtil.memPutByte(staging.mapped + i, (byte)(pix & 0xFF));              //R
                        MemoryUtil.memPutByte(staging.mapped + i + 1, (byte)((pix >> 8) & 0xFF));   //G
                        MemoryUtil.memPutByte(staging.mapped + i + 1, (byte)((pix >> 16) & 0xFF));  //B
                        MemoryUtil.memPutByte(staging.mapped + i + 1, (byte)((pix >> 24) & 0xFF));  //A
                        i += 4;
                    }
                }
            }
        }
        
        staging.flush(0, totalBytes);
        
        RtImage Image = ctx.create1Texture3D(size, size, slices, format); //创建3D纹理，1byte/像素，灰度图， layout 暂时为 UNDEFINED
        ctx.create2Texture3D(size, size, slices, staging, Image.image); // staging 拷贝到 noiseImage.image （GPU侧的3D纹理），并把纹理的 layout 从 UNDEFINED 转为 SHADER_READ_ONLY_OPTIMAL
        
        staging.destroy(); //销毁 staging，数据传入gpu后就不需要了
        return Image.view;
    }
}