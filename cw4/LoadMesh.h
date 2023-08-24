#pragma once
#include"Header.h"

#include <tuple>

struct LoadMeshInfo
{
	uint32_t materialIndex;
	lut::Buffer loadIndices;
	lut::Buffer loadData;

	std::size_t loadIndexCount;
};



struct LoadModel
{
	std::vector<LoadMeshInfo>meshInfo;
	std::vector<VkDescriptorSet>descriptorInfo;
	std::vector<lut::Image>images;
	std::vector<lut::ImageView> imageViews;
};

LoadModel create_loaded_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, VkDescriptorPool& dpool, VkCommandPool& aCmdPool, VkDescriptorSetLayout& aObjectLayout, VkSampler& aSampler, BakedModel const& ponzaModel);