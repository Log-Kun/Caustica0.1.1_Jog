package dev.comfyfluffy.caustica.rt;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;

import com.mojang.blaze3d.vertex.PoseStack;

import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.terrain.RtTerrain;
import net.minecraft.world.level.Level;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.state.BlockState;

import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;

import java.util.HashMap;
import java.util.Map;

import net.minecraft.world.entity.player.*;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.Items;

import net.minecraft.client.Minecraft;
import net.minecraft.client.player.AbstractClientPlayer;
import net.minecraft.client.renderer.entity.*;
import net.minecraft.client.renderer.entity.player.AvatarRenderer;
import net.minecraft.client.model.*;
import net.minecraft.world.phys.Vec3;


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

    private final Map<Block, float[]> LIGHT_COLORS = new HashMap<>(); // 方块 -> 光源颜色：色调(官方 Vibrant Visuals) x (发光等级/15)
    { //实例初始化语块，没有大括号则会被视为普通语句，非法
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
        //LIGHT_COLORS.put(Blocks.LAVA, new float[] { 0.83f, 0.52f, 0.17f }); // 琥珀 #d3852b, Lv15
        LIGHT_COLORS.put(Blocks.FIRE, new float[] { 0.95f, 0.60f, 0.37f }); // 火焰橙 #f39a5e, Lv15
        LIGHT_COLORS.put(Blocks.SOUL_FIRE, new float[] { 0.20f, 0.53f, 0.67f }); // 灵魂青, Lv10
        //LIGHT_COLORS.put(Blocks.MAGMA_BLOCK, new float[] { 0.17f, 0.10f, 0.03f }); // 琥珀 #d3852b, Lv3
        LIGHT_COLORS.put(Blocks.BEACON, new float[] { 0.60f, 0.90f, 1.00f }); // 青白(官方默认白), Lv15
        //LIGHT_COLORS.put(Blocks.CONDUIT, new float[] { 0.40f, 0.80f, 1.00f }); // 青(官方默认白), Lv15
        //LIGHT_COLORS.put(Blocks.GLOW_LICHEN, new float[] { 0.066f, 0.088f, 0.068f }); // 苔绿 #b7f1bc, Lv1
        //LIGHT_COLORS.put(Blocks.CRYING_OBSIDIAN, new float[] { 0.33f, 0.13f, 0.53f }); // 紫, Lv10
        //LIGHT_COLORS.put(Blocks.RESPAWN_ANCHOR, new float[] { 0.64f, 0.20f, 0.92f }); // 紫 #a233eb, 满充能 Lv15
        //LIGHT_COLORS.put(Blocks.ENDER_CHEST, new float[] { 0.23f, 0.09f, 0.37f }); // 紫, Lv7
        LIGHT_COLORS.put(Blocks.CANDLE, new float[] { 0.17f, 0.10f, 0.03f }); // 琥珀 #d3852b, 单支 Lv3
        //LIGHT_COLORS.put(Blocks.FIREFLY_BUSH, new float[] { 0.12f, 0.10f, 0.08f }); // 暖铜 #e8c398, Lv2(未积雪)
    }

    private int frameCounter = 0;

    List<PointLight> lights = new ArrayList<>(); //新建数组存储光源坐标
    //获取坐标list

    public List<PointLight> scanNearbyLights(Level level, BlockPos centre) {
        frameCounter = (frameCounter + 1) & 31; // 自增，始终在 0~31 循环。& 31 等价于 % 32 但是更快
        

        int nowh = frameCounter - 16;
        lights.removeIf(element -> element.y == centre.getY() + nowh // 移除这一层的信息，以进行更新
                || element.y < centre.getY() - 16 // 后面防止玩家上下平移时列表爆炸
                || element.y > centre.getY() + 15);

        for (BlockPos pos : BlockPos.betweenClosed(centre.offset(64, nowh, 64), centre.offset(-64, nowh, -64))) { // 范围内扫描 冒号遍历是语法糖
            Block block_info = level.getBlockState(pos).getBlock();
            float[] color = LIGHT_COLORS.getOrDefault(block_info, new float[] { -1.0f, -1.0f, -1.0f }); // 可以用get，但返回null就炸了
            if (color[0] >= 0) {
                lights.add(new PointLight(pos.getX(), pos.getY(), pos.getZ(), color[0], color[1], color[2]));
            }
        }
        return lights;
    }




    //传入GPU侧
    public static RtBuffer lightBuffer;  // GPU 上的缓冲区，其中 mapped 是CPU能写的地址，deviceAddress 是GPU可读的地址
    private static final int MAX_LIGHTS = 1024;  // 最多支持1024个光源
    public int currentLightCount = 0;  // 当前帧实际有多少个光源

    public void initialize_Lights(RtContext ctx) {
        lightBuffer = ctx.createBuffer(
            (long) MAX_LIGHTS * 32,                   //大小 每个光源占 32 字节
            VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  // 用途：存储缓冲区
            true,                        // host-visible，CPU能写
            "rt point lights"                  // 调试名字
        );
    }

    public void updateLights(Level level, BlockPos center) {

        List<PointLight> lights = scanNearbyLights(level, center);

        if(frameCounter == 0){ //仅在某一帧更新光源，防止列表更新导致闪烁
            
            ByteBuffer buffer = MemoryUtil.memByteBuffer(lightBuffer.mapped, (int) lightBuffer.size); //拿到缓冲区的 CPU 可写内存
            
            long addr = lightBuffer.mapped;
            currentLightCount = Math.min(lights.size(), MAX_LIGHTS);  // 不超过容量
            RtTerrain terrain = RtTerrain.currentOrNull();
            if (terrain != null){
                for (int i = 0; i < currentLightCount; i++) {
                    PointLight l = lights.get(i);
                    MemoryUtil.memPutInt(addr + 0, l.x - terrain.blockX); //自动在内存对齐位写入，主要顾虑在slang的结构体
                    MemoryUtil.memPutInt(addr + 4, l.y - terrain.blockY);
                    MemoryUtil.memPutInt(addr + 8, l.z - terrain.blockZ);
                    MemoryUtil.memPutFloat(addr + 16, l.r);
                    MemoryUtil.memPutFloat(addr + 20, l.g);
                    MemoryUtil.memPutFloat(addr + 24, l.b);
                    addr += 32; //需要对齐到16的倍数
                }
                
                lightBuffer.flush(0L, (long) currentLightCount * 32); //告诉 Vulkan：我写完了，数据可以从 CPU 传到 GPU 了
            }
        }
    }



    //手持光源部分
    private final Map<Item, float[]> HandLightMap = new HashMap<>();
    {
        HandLightMap.put(Items.TORCH, new float[]{0.847f, 0.784f, 0.542f});
        HandLightMap.put(Items.SOUL_TORCH, new float[]{0.0f, 0.559f, 0.559f});
        HandLightMap.put(Items.REDSTONE_TORCH, new float[]{0.327f, 0.0f, 0.0f});
        HandLightMap.put(Items.COPPER_TORCH, new float[]{0.156f, 0.847f, 0.156f});
        HandLightMap.put(Items.LANTERN, new float[]{0.81f, 0.51f, 0.20f});
        HandLightMap.put(Items.SOUL_LANTERN, new float[]{0.300f, 0.633f, 0.640f});
        HandLightMap.put(Items.GLOWSTONE, new float[]{1.0f, 0.91f, 0.64f});
        HandLightMap.put(Items.SEA_LANTERN, new float[]{0.65f, 0.91f, 1.0f});
        HandLightMap.put(Items.SHROOMLIGHT, new float[]{0.6f, 0.2f, 0.2f});
        HandLightMap.put(Items.JACK_O_LANTERN, new float[]{1.0f, 0.60f, 0.20f});
        HandLightMap.put(Items.REDSTONE_LAMP, new float[]{1.0f, 0.55f, 0.25f});
        HandLightMap.put(Items.CAMPFIRE, new float[]{0.81f, 0.37f, 0.22f});
        HandLightMap.put(Items.SOUL_CAMPFIRE, new float[]{0.0f, 0.559f, 0.559f});
        HandLightMap.put(Items.END_ROD, new float[]{0.847f, 0.847f, 0.847f});
        HandLightMap.put(Items.OCHRE_FROGLIGHT, new float[]{1.0f, 0.91f, 0.64f});
        HandLightMap.put(Items.VERDANT_FROGLIGHT, new float[]{0.5f, 0.65f, 0.59f});
        HandLightMap.put(Items.PEARLESCENT_FROGLIGHT, new float[]{0.95f, 0.5f, 0.65f});
    }
    public static RtBuffer HandLightBuffer;

    public void initialize_HandLight(RtContext ctx) {
        HandLightBuffer = ctx.createBuffer(
            (long) 32,
            VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            true,
            "rt hand lights"
        );
    }



    public void updateHandLight(LocalPlayer player) {
        Vec3 pos = player.getEyePosition();
        Vec3 view = player.getViewVector(1); //不知道这数值有啥用
        RtTerrain terrain = RtTerrain.currentOrNull();
        Item iteminfo = player.getMainHandItem().getItem();
        float[] color = HandLightMap.getOrDefault(iteminfo, new float[]{0.0f, 0.0f, 0.0f});
        long addr = HandLightBuffer.mapped; //Java要写，用mapped
        MemoryUtil.memPutFloat(addr, (float)(pos.x - terrain.blockX + view.x * 0.4));
        MemoryUtil.memPutFloat(addr + 4, (float)(pos.y - terrain.blockY + view.y * 0.4));
        MemoryUtil.memPutFloat(addr + 8, (float)(pos.z - terrain.blockZ + view.z * 0.4));
        MemoryUtil.memPutFloat(addr + 16, color[0]); //对齐到2的整数次幂
        MemoryUtil.memPutFloat(addr + 20, color[1]);
        MemoryUtil.memPutFloat(addr + 24, color[2]);
        HandLightBuffer.flush(0L, (long) 32);
    }
}