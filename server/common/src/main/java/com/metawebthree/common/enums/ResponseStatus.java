package com.metawebthree.common.enums;

public enum ResponseStatus {
    SUCCESS("0000", "操作成功"),

    PARAM_VALIDATION_ERROR("1001", "参数校验失败"),
    PARAM_MISSING_ERROR("1002", "参数缺失"),
    PARAM_TYPE_ERROR("1003", "参数类型错误"),

    USER_NOT_FOUND("2001", "用户不存在"),
    USER_PASSWORD_ERROR("2002", "密码错误"),
    USER_TOKEN_EXPIRED("2003", "Token已过期"),
    USER_TOKEN_INVALID("2004", "Token无效"),
    USER_WALLET_MISMATCH("2005", "钱包地址不匹配"),

    PRODUCT_NOT_FOUND("3001", "商品不存在"),
    PRODUCT_OFF_SHELF("3002", "商品已下架"),
    PRODUCT_CREATE_FAILED("3003", "商品创建失败"),
    PRODUCT_UPDATE_FAILED("3004", "商品更新失败"),
    PRODUCT_DELETE_FAILED("3005", "商品删除失败"),
    PRODUCT_IMAGE_UPLOAD_FAILED("3006", "商品图片上传失败"),

    ORDER_NOT_FOUND("4001", "订单不存在"),
    ORDER_CREATE_FAILED("4002", "订单创建失败"),
    ORDER_CANCEL_FAILED("4003", "订单取消失败"),
    ORDER_STATUS_INVALID("4004", "订单状态不允许此操作"),

    PAYMENT_INSUFFICIENT_BALANCE("5001", "余额不足"),
    PAYMENT_EXCHANGE_FAILED("5002", "兑换失败"),
    PAYMENT_PRICE_NOT_FOUND("5003", "价格不存在"),

    MEDIA_FILE_NOT_FOUND("6001", "文件不存在"),
    MEDIA_UPLOAD_FAILED("6002", "文件上传失败"),
    MEDIA_DELETE_FAILED("6003", "文件删除失败"),

    SYSTEM_ERROR("9999", "系统错误");

    private final String code;
    private final String message;

    ResponseStatus(String code, String message) {
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
