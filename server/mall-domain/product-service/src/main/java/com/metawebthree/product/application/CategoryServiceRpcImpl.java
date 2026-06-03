package com.metawebthree.product.application;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.beans.factory.annotation.Autowired;

import com.metawebthree.common.generated.rpc.CategoryDTO;
import com.metawebthree.common.generated.rpc.CategoryNodeDTO;
import com.metawebthree.common.generated.rpc.CategoryService;
import com.metawebthree.common.generated.rpc.GetCategoryByIdRequest;
import com.metawebthree.common.generated.rpc.GetCategoryByIdResponse;
import com.metawebthree.common.generated.rpc.GetCategoriesRequest;
import com.metawebthree.common.generated.rpc.GetCategoriesResponse;
import com.metawebthree.common.generated.rpc.GetCategoryTreeRequest;
import com.metawebthree.common.generated.rpc.GetCategoryTreeResponse;
import com.metawebthree.product.domain.model.ProductCategory;
import com.metawebthree.product.interfaces.web.dto.ProductCategoryNode;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@DubboService
public class CategoryServiceRpcImpl implements CategoryService {

    @Autowired
    private ProductCategoryApplicationService categoryService;

    @Override
    public GetCategoryByIdResponse getCategoryById(GetCategoryByIdRequest request) {
        log.info("Dubbo RPC: getCategoryById called with id: {}", request.getId());
        
        try {
            List<ProductCategory> categories = categoryService.findSubCategories(request.getId());
            // findSubCategories returns subcategories, we need to find by id directly
            // For now, return empty - need to add findById to the application service
            return GetCategoryByIdResponse.newBuilder().build();
        } catch (Exception e) {
            log.error("Failed to get category by id: {}, error: {}", request.getId(), e.getMessage());
            return GetCategoryByIdResponse.newBuilder().build();
        }
    }

    @Override
    public CompletableFuture<GetCategoryByIdResponse> getCategoryByIdAsync(GetCategoryByIdRequest request) {
        return CompletableFuture.completedFuture(getCategoryById(request));
    }

    @Override
    public GetCategoriesResponse getCategories(GetCategoriesRequest request) {
        log.info("Dubbo RPC: getCategories called with page: {}, size: {}", request.getPage(), request.getSize());
        
        try {
            List<ProductCategory> allCategories = categoryService.findSubCategories(null);
            if (allCategories == null) {
                allCategories = List.of();
            }
            
            List<CategoryDTO> categoryDTOs = allCategories.stream()
                    .map(this::toCategoryDTO)
                    .collect(Collectors.toList());
            
            return GetCategoriesResponse.newBuilder()
                    .addAllCategories(categoryDTOs)
                    .setTotalCount(categoryDTOs.size())
                    .build();
        } catch (Exception e) {
            log.error("Failed to get categories, error: {}", e.getMessage());
            return GetCategoriesResponse.newBuilder()
                    .setTotalCount(0)
                    .build();
        }
    }

    @Override
    public CompletableFuture<GetCategoriesResponse> getCategoriesAsync(GetCategoriesRequest request) {
        return CompletableFuture.completedFuture(getCategories(request));
    }

    @Override
    public GetCategoryTreeResponse getCategoryTree(GetCategoryTreeRequest request) {
        log.info("Dubbo RPC: getCategoryTree called");
        
        try {
            List<ProductCategoryNode> tree = categoryService.categoryTreeList();
            List<CategoryNodeDTO> treeDTOs = tree.stream()
                    .map(this::toCategoryNodeDTO)
                    .collect(Collectors.toList());
            
            return GetCategoryTreeResponse.newBuilder()
                    .addAllTree(treeDTOs)
                    .build();
        } catch (Exception e) {
            log.error("Failed to get category tree, error: {}", e.getMessage());
            return GetCategoryTreeResponse.newBuilder().build();
        }
    }

    @Override
    public CompletableFuture<GetCategoryTreeResponse> getCategoryTreeAsync(GetCategoryTreeRequest request) {
        return CompletableFuture.completedFuture(getCategoryTree(request));
    }

    private CategoryDTO toCategoryDTO(ProductCategory category) {
        return CategoryDTO.newBuilder()
                .setId(category.getId())
                .setName(category.getName() != null ? category.getName() : "")
                .setParentId(category.getParentId() != null ? category.getParentId() : 0)
                .setSortOrder(category.getSort() != null ? category.getSort() : 0)
                .setIcon(category.getIcon() != null ? category.getIcon() : "")
                .setEnabled(category.isDisplayedInNav())
                .build();
    }

    private CategoryNodeDTO toCategoryNodeDTO(ProductCategoryNode node) {
        CategoryNodeDTO.Builder builder = CategoryNodeDTO.newBuilder()
                .setId(node.getId())
                .setName(node.getName() != null ? node.getName() : "")
                .setParentId(node.getParentId() != null ? node.getParentId() : 0)
                .setSortOrder(node.getSort() != null ? node.getSort() : 0)
                .setEnabled(node.isDisplayedInNav());
        
        if (node.getIcon() != null) {
            builder.setIcon(node.getIcon());
        }
        
        if (node.getChildren() != null) {
            builder.addAllChildren(node.getChildren().stream()
                    .map(this::toCategoryNodeDTO)
                    .collect(Collectors.toList()));
        }
        
        return builder.build();
    }
}