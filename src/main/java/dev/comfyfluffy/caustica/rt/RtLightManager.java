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
    int x, y, z;
    float r, g, b;
    PointLight(int x, int y, int z,float r, float g, float b){
        this.x = x;
        this.y = y;
        this.z = z;
        this.r = r;
        this.g = g;
        this.b = b;
    }
}

public class RtLightManager {
    private static final Map<Block, float[]> LIGHT_COLORS = new HashMap<>(); // 方块 -> 光源颜色：色调(官方 Vibrant Visuals) x (发光等级/15)
    static {
        LIGHT_COLORS.put(Blocks.TORCH, new float[] { 0.89f, 0.56f, 0.34f }); // 火焰橙 #f39a5e, Lv14
        LIGHT_COLORS.put(Blocks.WALL_TORCH, new float[] { 0.89f, 0.56f, 0.34f }); // 火焰橙, Lv14
        LIGHT_COLORS.put(Blocks.SOUL_TORCH, new float[] { 0.13f, 0.53f, 0.67f }); // 灵魂青, Lv10
        LIGHT_COLORS.put(Blocks.SOUL_WALL_TORCH, new float[] { 0.13f, 0.53f, 0.67f }); // 灵魂青, Lv10
        LIGHT_COLORS.put(Blocks.REDSTONE_TORCH, new float[] { 0.36f, 0.08f, 0.06f }); // 红石红 #c32c1f, Lv7
        LIGHT_COLORS.put(Blocks.REDSTONE_WALL_TORCH, new float[] { 0.36f, 0.08f, 0.06f }); // 红石红, Lv7
        LIGHT_COLORS.put(Blocks.CAMPFIRE, new float[] { 0.95f, 0.60f, 0.37f }); // 火焰橙, Lv15
        LIGHT_COLORS.put(Blocks.SOUL_CAMPFIRE, new float[] { 0.14f, 0.57f, 0.60f }); // 灵魂青 #36d9e6, Lv10
        LIGHT_COLORS.put(Blocks.LANTERN, new float[] { 0.95f, 0.60f, 0.37f }); // 火焰橙, Lv15
        LIGHT_COLORS.put(Blocks.SOUL_LANTERN, new float[] { 0.14f, 0.57f, 0.60f }); // 灵魂青, Lv10
        LIGHT_COLORS.put(Blocks.REDSTONE_LAMP, new float[] { 0.91f, 0.76f, 0.60f }); // 暖铜 #e8c398, Lv15
        LIGHT_COLORS.put(Blocks.SEA_LANTERN, new float[] { 0.69f, 0.85f, 0.83f }); // 海晶青 #b0dad3, Lv15
        LIGHT_COLORS.put(Blocks.JACK_O_LANTERN, new float[] { 0.83f, 0.52f, 0.17f }); // 琥珀 #d3852b, Lv15
        //LIGHT_COLORS.put(Blocks.COPPER_BULB, new float[] { 0.91f, 0.76f, 0.60f }); // 暖铜 #e8c398, Lv15
        LIGHT_COLORS.put(Blocks.GLOWSTONE, new float[] { 1.00f, 0.90f, 0.50f }); // 暖黄(官方默认白), Lv15
        LIGHT_COLORS.put(Blocks.SHROOMLIGHT, new float[] { 0.83f, 0.52f, 0.17f }); // 琥珀 #d3852b, Lv15
        LIGHT_COLORS.put(Blocks.OCHRE_FROGLIGHT, new float[] { 0.98f, 0.94f, 0.65f }); // 赭黄 #f9efa5, Lv15
        LIGHT_COLORS.put(Blocks.VERDANT_FROGLIGHT, new float[] { 0.72f, 0.95f, 0.74f }); // 翠绿 #b7f1bc, Lv15
        LIGHT_COLORS.put(Blocks.PEARLESCENT_FROGLIGHT, new float[] { 0.85f, 0.45f, 0.87f }); // 珠光粉紫 #da73de, Lv15
        LIGHT_COLORS.put(Blocks.SEA_PICKLE, new float[] { 0.28f, 0.34f, 0.33f }); // 海晶青 #b0dad3, 单株 Lv6
        LIGHT_COLORS.put(Blocks.END_ROD, new float[] { 0.80f, 0.42f, 0.81f }); // 粉紫 #da73de, Lv14
        LIGHT_COLORS.put(Blocks.LAVA, new float[] { 0.83f, 0.52f, 0.17f }); // 琥珀 #d3852b, Lv15
        LIGHT_COLORS.put(Blocks.FIRE, new float[] { 0.95f, 0.60f, 0.37f }); // 火焰橙 #f39a5e, Lv15
        LIGHT_COLORS.put(Blocks.SOUL_FIRE, new float[] { 0.20f, 0.53f, 0.67f }); // 灵魂青, Lv10
        LIGHT_COLORS.put(Blocks.MAGMA_BLOCK, new float[] { 0.17f, 0.10f, 0.03f }); // 琥珀 #d3852b, Lv3
        LIGHT_COLORS.put(Blocks.BEACON, new float[] { 0.60f, 0.90f, 1.00f }); // 青白(官方默认白), Lv15
        LIGHT_COLORS.put(Blocks.CONDUIT, new float[] { 0.40f, 0.80f, 1.00f }); // 青(官方默认白), Lv15
        LIGHT_COLORS.put(Blocks.GLOW_LICHEN, new float[] { 0.33f, 0.44f, 0.34f }); // 苔绿 #b7f1bc, Lv7
        LIGHT_COLORS.put(Blocks.CRYING_OBSIDIAN, new float[] { 0.33f, 0.13f, 0.53f }); // 紫, Lv10
        LIGHT_COLORS.put(Blocks.RESPAWN_ANCHOR, new float[] { 0.64f, 0.20f, 0.92f }); // 紫 #a233eb, 满充能 Lv15
        LIGHT_COLORS.put(Blocks.ENDER_CHEST, new float[] { 0.23f, 0.09f, 0.37f }); // 紫, Lv7
        LIGHT_COLORS.put(Blocks.CANDLE, new float[] { 0.17f, 0.10f, 0.03f }); // 琥珀 #d3852b, 单支 Lv3
        LIGHT_COLORS.put(Blocks.FIREFLY_BUSH, new float[] { 0.12f, 0.10f, 0.08f }); // 暖铜 #e8c398, Lv2(未积雪)
    }

    public int frameCounter = 0;

    List<PointLight> lights = new ArrayList<>(); //新建数组存储光源坐标
    //获取坐标list

    public List<PointLight> scanNearbyLights(Level level, BlockPos centre) {
        frameCounter = (frameCounter + 1) & 31; // 自增，始终在 0~31 循环。& 31 等价于 % 32 但是更快
        

        int nowh = frameCounter - 16;
        lights.removeIf(element -> element.y == centre.getY() + nowh // 移除这一层的信息，以进行更新
                || element.y < centre.getY() - 33 // 后面防止玩家上下平移时列表爆炸
                || element.y > centre.getY() + 33);

        for (BlockPos pos : BlockPos.betweenClosed(centre.offset(32, nowh, 32), centre.offset(-32, nowh, -32))) { // 范围内扫描 冒号遍历是语法糖
            BlockState state = level.getBlockState(pos); // level 类里 getBlockState 方法返回方块状态
            int L = state.getLightEmission(); // BlockState 类里 getLightEmission 方法返回方块亮度
            if (L > 0) {
                Block block_info = state.getBlock();
                float[] color = LIGHT_COLORS.getOrDefault(block_info, new float[] { 0.05f, 0.05f, 0.05f }); // 可以用get，但返回null就炸了
                lights.add(new PointLight(pos.getX(), pos.getY(), pos.getZ(), color[0], color[1], color[2]));
            }
        }
        return lights;
    }




    //传入GPU侧
    public RtBuffer lightBuffer;  // GPU 上的缓冲区，其中 mapped 是CPU能写的地址，deviceAddress 是GPU可读的地址
    private static final int MAX_LIGHTS = 1024;  // 最多支持8192个光源
    public int currentLightCount = 0;  // 当前帧实际有多少个光源

    public void init(RtContext ctx) {
        lightBuffer = ctx.createBuffer(
            (long) MAX_LIGHTS * 32,                   //大小 每个光源占 32 字节
            VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  // 用途：存储缓冲区
            true,                        // host-visible，CPU能写
            "rt point lights"                  // 调试名字
        );
    }
    public void updateLights(Level level, BlockPos center) {

        // 1. 扫描附近光源
        List<PointLight> lights = scanNearbyLights(level, center);

        if(frameCounter == 0){ //仅在某一帧更新光源

            
            // 2. 拿到缓冲区的 CPU 可写内存
            ByteBuffer buffer = MemoryUtil.memByteBuffer(lightBuffer.mapped, (int) lightBuffer.size);
            
            // 3. 逐个写入
            long addr = lightBuffer.mapped;
            int count = Math.min(lights.size(), MAX_LIGHTS);  // 不超过容量
            RtTerrain terrain = RtTerrain.currentOrNull();
            //if (terrain == null)return lights;
            for (int i = 0; i < count; i++) {
                PointLight l = lights.get(i);
                long base = addr + i * 32; //需要对齐到16的倍数
                MemoryUtil.memPutInt(base + 0, l.x - terrain.blockX); //自动在内存对齐位写入，主要顾虑在slang的结构体
                MemoryUtil.memPutInt(base + 4, l.y - terrain.blockY);
                MemoryUtil.memPutInt(base + 8, l.z - terrain.blockZ);
                MemoryUtil.memPutFloat(base + 16, l.r);
                MemoryUtil.memPutFloat(base + 20, l.g);
                MemoryUtil.memPutFloat(base + 24, l.b);
            }
            
            // 4. 告诉 Vulkan：我写完了，数据可以从 CPU 传到 GPU 了
            lightBuffer.flush(0L, (long) count * 32);
            
            // 5. 记住这一帧有多少个光源
            currentLightCount = count;
        }
    }
}