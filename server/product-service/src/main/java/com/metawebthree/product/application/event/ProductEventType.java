package com.metawebthree.product.event;

/**
 * 产品领域事件类型枚举
 * 所有产品相关的事件必须通过此枚举定义，禁止硬编码字符串
 */
public enum ProductEventType {
    PRODUCT_CREATED("product.created"),
    PRODUCT_UPDATED("product.updated"),
    PRODUCT_DELETED("product.deleted"),
    PRODUCT_IMAGE_UPLOADED("product.image.uploaded");

    private final String eventName;

    ProductEventType(String eventName) {
        this.eventName = eventName;
    }

    public String getEventName() {
        return eventName;
    }
}
