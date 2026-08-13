package com.metawebthree.recommendation.infrastructure.aishopping.product;

import com.metawebthree.common.generated.rpc.ListProductsRequest;
import com.metawebthree.common.generated.rpc.ListProductsResponse;
import com.metawebthree.common.generated.rpc.ProductDetailProto;
import com.metawebthree.common.generated.rpc.ProductService;
import java.util.ArrayList;
import java.util.List;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

/**
 * Fetches product data (name/image/price) from product-service over Dubbo RPC
 * for index building and matching.
 */
@Component
public class ProductDataProvider {

    private static final int PAGE_SIZE = 100;
    private static final int MAX_PAGES = 500;

    @DubboReference(check = false, lazy = true)
    private ProductService productService;

    public List<ProductItem> fetchAllProducts() {
        List<ProductItem> items = new ArrayList<>();
        for (int page = 1; page <= MAX_PAGES; page++) {
            ListProductsRequest request = ListProductsRequest.newBuilder()
                    .setPage(page)
                    .setSize(PAGE_SIZE)
                    .build();
            ListProductsResponse response = productService.listProducts(request);
            if (response == null || response.getProductsList().isEmpty()) {
                break;
            }
            for (ProductDetailProto product : response.getProductsList()) {
                items.add(from(product));
            }
            if (response.getProductsList().size() < PAGE_SIZE) {
                break;
            }
        }
        return items;
    }

    private ProductItem from(ProductDetailProto product) {
        ProductItem item = new ProductItem();
        item.id = product.getId();
        item.name = product.getName();
        item.sku = product.getSku();
        item.pic = product.getPic();
        item.subTitle = product.getSubTitle();
        item.price = product.getPrice();
        item.categoryId = product.getCategoryId();
        item.pictures = new ArrayList<>(product.getPicturesList());
        item.description = product.getDescription();
        return item;
    }

    public static class ProductItem {
        public long id;
        public String name;
        public String sku;
        public String pic;
        public String subTitle;
        public double price;
        public long categoryId;
        public List<String> pictures;
        public String description;
    }
}
