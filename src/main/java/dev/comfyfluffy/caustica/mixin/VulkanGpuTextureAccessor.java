package dev.comfyfluffy.caustica.mixin;

import com.mojang.blaze3d.vulkan.VulkanGpuTexture;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(VulkanGpuTexture.class)
public interface VulkanGpuTextureAccessor {
	@Accessor("vkImage")
	long caustica$getVkImage();
}
