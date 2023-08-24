#include "LoadMesh.h"
#include"Header.h"

#include <utility>

#include <cstdio>
#include <cassert>
namespace lut = labutils;
LoadModel create_loaded_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, VkDescriptorPool& dpool, VkCommandPool& aCmdPool, VkDescriptorSetLayout& aObjectLayout, VkSampler& aSampler, BakedModel const& ponzaModel)
{
	//final return
	LoadModel loadedModel;

	for (auto& mesh : ponzaModel.meshes)
	{

		std::vector<float>vertices;
		std::vector<uint32_t> indices;

		for (size_t j = 0; j < mesh.positions.size(); j++)
		{
			vertices.emplace_back(mesh.positions[j].x);
			vertices.emplace_back(mesh.positions[j].y);
			vertices.emplace_back(mesh.positions[j].z);

			vertices.emplace_back(mesh.texcoords[j].x);
			vertices.emplace_back(mesh.texcoords[j].y);

			vertices.emplace_back(mesh.normals[j].x);
			vertices.emplace_back(mesh.normals[j].y);
			vertices.emplace_back(mesh.normals[j].z);
		}

		indices = mesh.indices;


		lut::Buffer vertexGPU = lut::create_buffer(
			aAllocator,
			sizeof(float) * vertices.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);


		lut::Buffer vertexStaging = lut::create_buffer(
			aAllocator,
			sizeof(float) * vertices.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);


		void* vertexPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, vertexStaging.allocation, &vertexPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(vertexPtr, vertices.data(), sizeof(float) * vertices.size());
		vmaUnmapMemory(aAllocator.allocator, vertexStaging.allocation);

		// Make sure the vulkan instance is still alive after all transfers are completed.
		lut::Fence uploadComplete = lut::create_fence(aContext);

		// Queue data uploads from staging buffers to the final buffers
		lut::CommandPool uploadPool = lut::create_command_pool(aContext);
		VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aContext, uploadPool.handle);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo);
			VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		VkBufferCopy verCopy{};
		verCopy.size = sizeof(float) * vertices.size();

		vkCmdCopyBuffer(uploadCmd, vertexStaging.buffer, vertexGPU.buffer,
			1, &verCopy);

		lut::buffer_barrier(uploadCmd,
			vertexGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Submit transfer commands
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1,
			&submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("(create mesh)Submitting commands\n"
				"vkQueueSubmit returned() %s", lut::to_string(res).c_str());
		}

		// Wait for command to finish before we destroy the temporary resources
		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle,
			VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n"
				"vkWaitForFence() returned %s", lut::to_string(res).c_str());
		}

		lut::Buffer indexGPU = lut::create_buffer(
			aAllocator,
			mesh.indices.size() * sizeof(uint32_t),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);
		lut::Buffer indexStaging = lut::create_buffer(
			aAllocator,
			mesh.indices.size() * sizeof(uint32_t),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);
		void* indexPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, indexStaging.allocation, &indexPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(indexPtr, mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t));
		vmaUnmapMemory(aAllocator.allocator, indexStaging.allocation);

		// Make sure the vulkan instance is still alive after all transfers are completed.
		lut::Fence indexComplete = lut::create_fence(aContext);

		// Queue data uploads from staging buffers to the final buffers
		lut::CommandPool indexPool = lut::create_command_pool(aContext);
		VkCommandBuffer indexCmd = lut::alloc_command_buffer(aContext, indexPool.handle);

		VkCommandBufferBeginInfo indexbeginInfo{};
		indexbeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		indexbeginInfo.flags = 0;
		indexbeginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(indexCmd, &indexbeginInfo);
			VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		VkBufferCopy indexCopy{};
		indexCopy.size = mesh.indices.size() * sizeof(uint32_t);

		vkCmdCopyBuffer(indexCmd, indexStaging.buffer, indexGPU.buffer,
			1, &indexCopy);

		lut::buffer_barrier(indexCmd,
			indexGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		if (auto const res = vkEndCommandBuffer(indexCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}
		VkSubmitInfo indexsubmitInfo{};
		indexsubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		indexsubmitInfo.commandBufferCount = 1;
		indexsubmitInfo.pCommandBuffers = &indexCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &indexsubmitInfo, indexComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n" "vkQueueSubmit() returned %s", lut::to_string(res));
		}

		if (auto const res = vkWaitForFences(aContext.device, 1, &indexComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());
		}
		LoadMeshInfo meshess;
		meshess.loadData = std::move(vertexGPU);
		meshess.loadIndexCount = static_cast<uint32_t>(mesh.indices.size());
		meshess.loadIndices = std::move(indexGPU);
		meshess.materialIndex = mesh.materialId;

		loadedModel.meshInfo.emplace_back(std::move(meshess));
	}


	for (size_t i = 0; i < ponzaModel.textures.size(); i++)
	{
		lut::Image image = lut::load_image_texture2d(ponzaModel.textures[i].path.c_str(), aContext, aCmdPool, aAllocator);
		lut::ImageView view = lut::create_image_view_texture2d(aContext, image.image, VK_FORMAT_R8G8B8A8_SRGB);

		loadedModel.images.emplace_back(std::move(image));
		loadedModel.imageViews.emplace_back(std::move(view));
	}


	uint32_t matCount = static_cast<uint32_t>(ponzaModel.materials.size());

	for (uint32_t i = 0; i < matCount; i++)
	{
		VkDescriptorSet descSet = lut::alloc_desc_set(aContext, dpool, aObjectLayout);

		VkDescriptorImageInfo imageInfo[3]{};
		imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo[0].imageView = loadedModel.imageViews[ponzaModel.materials[i].baseColorTextureId].handle;
		imageInfo[0].sampler = aSampler;

		imageInfo[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo[1].imageView = loadedModel.imageViews[ponzaModel.materials[i].roughnessTextureId].handle;
		imageInfo[1].sampler = aSampler;

		imageInfo[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo[2].imageView = loadedModel.imageViews[ponzaModel.materials[i].metalnessTextureId].handle;
		imageInfo[2].sampler = aSampler;

		VkWriteDescriptorSet desc[3]{};
		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = descSet;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &imageInfo[0];

		desc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[1].dstSet = descSet;
		desc[1].dstBinding = 1;
		desc[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[1].descriptorCount = 1;
		desc[1].pImageInfo = &imageInfo[1];

		desc[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[2].dstSet = descSet;
		desc[2].dstBinding = 2;
		desc[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[2].descriptorCount = 1;
		desc[2].pImageInfo = &imageInfo[2];

		constexpr auto descNums = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(aContext.device, descNums, desc, 0, nullptr);
		loadedModel.descriptorInfo.emplace_back(std::move(descSet));
	}
	return loadedModel;
}
