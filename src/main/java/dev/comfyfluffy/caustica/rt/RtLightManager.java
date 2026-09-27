package dev.comfyfluffy.caustica.rt;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;

import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.terrain.RtTerrain;
import net.minecraft.world.level.Level;

import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.state.BlockState;

import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;

import java.util.HashMap;
import java.util.Map;

class PointLight {
    float x, y, z;
    float r, g, b;
    int L;
    PointLight(float x, float y, float z,float r, float g, float b, int L){
        this.x = x;
        this.y = y;
        this.z = z;
        this.r = r;
        this.g = g;
        this.b = b;
        this.L = L;
    }
}

public class RtLightManager {

    private static final Map<Block, float[]> LIGHT_COLORS = new HashMap<>(); //正常mat只能塞10项 final表示不可修改
    static {
        LIGHT_COLORS.put(Blocks.TORCH, new float[] { 1.0f, 0.9f, 0.6f });
        LIGHT_COLORS.put(Blocks.TORCH, new float[] { 1.0f, 0.9f, 0.6f }); // 暖黄
        LIGHT_COLORS.put(Blocks.WALL_TORCH, new float[] { 1.0f, 0.9f, 0.6f });
        LIGHT_COLORS.put(Blocks.SOUL_TORCH, new float[] { 0.2f, 0.8f, 1.0f }); // 青蓝
        LIGHT_COLORS.put(Blocks.SOUL_WALL_TORCH, new float[] { 0.2f, 0.8f, 1.0f });
        LIGHT_COLORS.put(Blocks.REDSTONE_TORCH, new float[] { 1.0f, 0.3f, 0.2f }); // 红
        LIGHT_COLORS.put(Blocks.REDSTONE_WALL_TORCH, new float[] { 1.0f, 0.3f, 0.2f });
        LIGHT_COLORS.put(Blocks.CAMPFIRE, new float[] { 1.0f, 0.7f, 0.3f }); // 橙黄
        LIGHT_COLORS.put(Blocks.SOUL_CAMPFIRE, new float[] { 0.3f, 0.7f, 1.0f });
        LIGHT_COLORS.put(Blocks.LANTERN, new float[] { 1.0f, 0.8f, 0.5f }); // 暖白
        LIGHT_COLORS.put(Blocks.SOUL_LANTERN, new float[] { 0.2f, 0.7f, 1.0f });
        LIGHT_COLORS.put(Blocks.REDSTONE_LAMP, new float[] { 1.0f, 0.5f, 0.2f }); // 橙红
        LIGHT_COLORS.put(Blocks.SEA_LANTERN, new float[] { 0.4f, 0.9f, 1.0f }); // 青蓝
        LIGHT_COLORS.put(Blocks.JACK_O_LANTERN, new float[] { 1.0f, 0.7f, 0.2f }); // 南瓜橙
        //LIGHT_COLORS.put(Blocks.COPPER_BULB, new float[] { 1.0f, 0.7f, 0.4f }); // 铜橙
        LIGHT_COLORS.put(Blocks.GLOWSTONE, new float[] { 1.0f, 0.9f, 0.5f }); // 暖黄
        LIGHT_COLORS.put(Blocks.SHROOMLIGHT, new float[] { 1.0f, 0.6f, 0.3f }); // 橙红
        LIGHT_COLORS.put(Blocks.OCHRE_FROGLIGHT, new float[] { 1.0f, 0.8f, 0.4f }); // 赭黄
        LIGHT_COLORS.put(Blocks.VERDANT_FROGLIGHT, new float[] { 0.5f, 1.0f, 0.5f }); // 翠绿
        LIGHT_COLORS.put(Blocks.PEARLESCENT_FROGLIGHT, new float[] { 1.0f, 0.8f, 1.0f }); // 珠光粉紫
        LIGHT_COLORS.put(Blocks.SEA_PICKLE, new float[] { 0.5f, 1.0f, 0.8f }); // 水绿
        LIGHT_COLORS.put(Blocks.END_ROD, new float[] { 1.0f, 1.0f, 0.9f }); // 冷白
        LIGHT_COLORS.put(Blocks.LAVA, new float[] { 1.0f, 0.5f, 0.1f }); // 亮橙
        LIGHT_COLORS.put(Blocks.FIRE, new float[] { 1.0f, 0.6f, 0.2f });
        LIGHT_COLORS.put(Blocks.SOUL_FIRE, new float[] { 0.3f, 0.8f, 1.0f });
        LIGHT_COLORS.put(Blocks.MAGMA_BLOCK, new float[] { 1.0f, 0.4f, 0.1f });
        LIGHT_COLORS.put(Blocks.BEACON, new float[] { 0.6f, 0.9f, 1.0f }); // 青白
        LIGHT_COLORS.put(Blocks.CONDUIT, new float[] { 0.4f, 0.8f, 1.0f });
        LIGHT_COLORS.put(Blocks.GLOW_LICHEN, new float[] { 0.5f, 1.0f, 0.5f }); // 绿
        LIGHT_COLORS.put(Blocks.CRYING_OBSIDIAN, new float[] { 0.5f, 0.2f, 0.8f }); // 紫
        LIGHT_COLORS.put(Blocks.RESPAWN_ANCHOR, new float[] { 0.6f, 0.2f, 1.0f }); // 紫
        LIGHT_COLORS.put(Blocks.ENDER_CHEST, new float[] { 0.5f, 0.2f, 0.8f });
        LIGHT_COLORS.put(Blocks.CANDLE, new float[] { 1.0f, 0.85f, 0.55f });
    }

    //获取坐标list
    public List<PointLight> scanNearbyLights(Level level, BlockPos centre, int radius){
        RtTerrain terrain = RtTerrain.currentOrNull();
        List<PointLight> lights = new ArrayList<>(); //新建数组存储光源坐标
        for(BlockPos pos : BlockPos.betweenClosed(centre.offset(-radius, -radius, -radius), centre.offset(radius, radius, radius))) { //范围内扫描 冒号遍历是语法糖
            BlockState state = level.getBlockState(pos); // level 类里 getBlockState 方法返回方块状态
            int L = state.getLightEmission(); // BlockState 类里 getLightEmission 方法返回方块亮度
            if(L > 0){
                Block block_info = state.getBlock();
                float[] color = LIGHT_COLORS.getOrDefault(block_info, new float[] { 1.0f, 1.0f, 1.0f }); //可以用get，但返回null就炸了
                lights.add(new PointLight(pos.getX() + 0.5f - terrain.blockX, pos.getY() + 0.5f - terrain.blockY, pos.getZ() + 0.5f - terrain.blockZ, color[0], color[1], color[2], L));
            }
        }
        return lights;
    }




    //传入GPU侧
    public RtBuffer lightBuffer;  // GPU 上的缓冲区，其中 mapped 是CPU能写的地址，deviceAddress 是GPU可读的地址
    private static final int MAX_LIGHTS = 256;  // 最多支持256个光源
    public int currentLightCount = 0;  // 当前帧实际有多少个光源

    public void init(RtContext ctx) {
        lightBuffer = ctx.createBuffer(
            (long) MAX_LIGHTS * 48,                   //大小 每个光源占 48 字节
            VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  // 用途：存储缓冲区
            true,                        // host-visible，CPU能写
            "rt point lights"                  // 调试名字
        );
    }
    public void updateLights(Level level, BlockPos center, int radius) {
        // 1. 扫描附近光源
        List<PointLight> lights = scanNearbyLights(level, center, radius);
        
        // 2. 拿到缓冲区的 CPU 可写内存
        ByteBuffer buffer = MemoryUtil.memByteBuffer(lightBuffer.mapped, (int) lightBuffer.size);
        
        // 3. 逐个写入
        long addr = lightBuffer.mapped;
        int count = Math.min(lights.size(), MAX_LIGHTS);  // 不超过容量
        for (int i = 0; i < count; i++) {
            PointLight l = lights.get(i);
            long base = addr + i * 48; //需要对齐到16的倍数
            MemoryUtil.memPutFloat(base + 0, l.x); //自动在内存对齐位写入，主要顾虑在slang的结构体
            MemoryUtil.memPutFloat(base + 4, l.y);
            MemoryUtil.memPutFloat(base + 8, l.z);
            MemoryUtil.memPutFloat(base + 16, l.r);
            MemoryUtil.memPutFloat(base + 20, l.g);
            MemoryUtil.memPutFloat(base + 24, l.b);
            MemoryUtil.memPutInt(base + 32, l.L);
        }
        
        // 4. 告诉 Vulkan：我写完了，数据可以从 CPU 传到 GPU 了
        lightBuffer.flush(0L, (long) count * 48);
        
        // 5. 记住这一帧有多少个光源
        currentLightCount = count;
    }
}