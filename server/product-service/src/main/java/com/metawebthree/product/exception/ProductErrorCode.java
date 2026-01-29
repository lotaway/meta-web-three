package com.metawebthree.product.exception;

public enum ProductErrorCode {
    PRODUCT_NOT_FOUND("PRODUCT_001", "Product not found"),
    PRODUCT_DELETE_FAILED("PRODUCT_002", "Failed to delete product"),
    PRODUCT_IMAGE_UPLOAD_FAILED("PRODUCT_003", "Failed to upload product image"),
    PRODUCT_CREATE_FAILED("PRODUCT_004", "Failed to create product"),
    PRODUCT_UPDATE_FAILED("PRODUCT_005", "Failed to update product");

    private final String code;
    private final String message;

    ProductErrorCode(String code, String message) {
        this.code = code;
        this.message = message;
    }

    public String getCode() {
        return code;
    }

    public String getMessage() {
        return message;
    }
}
