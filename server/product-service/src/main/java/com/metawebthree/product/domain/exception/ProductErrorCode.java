package com.metawebthree.product.exception;

public enum ProductErrorCode {
    NOT_FOUND("PRODUCT_001", "商品不存在"),
    DELETE_FAILED("PRODUCT_002", "商品删除失败"),
    IMAGE_UPLOAD_FAILED("PRODUCT_003", "商品图片上传失败"),
    CREATE_FAILED("PRODUCT_004", "商品创建失败"),
    UPDATE_FAILED("PRODUCT_005", "商品更新失败");

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
