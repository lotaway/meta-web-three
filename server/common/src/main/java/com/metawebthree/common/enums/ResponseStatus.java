package com.metawebthree.common.enums;

public enum ResponseStatus {
    SUCCESS("0000", "Operation successful"),

    PARAM_VALIDATION_ERROR("1001", "Parameter validation failed"),
    PARAM_MISSING_ERROR("1002", "Parameter missing"),
    PARAM_TYPE_ERROR("1003", "Parameter type error"),

    USER_NOT_FOUND("2001", "User not found"),
    USER_PASSWORD_ERROR("2002", "Password error"),
    USER_TOKEN_EXPIRED("2003", "Token expired"),
    USER_TOKEN_INVALID("2004", "Token invalid"),
    USER_WALLET_MISMATCH("2005", "Wallet address mismatch"),

    PRODUCT_NOT_FOUND("3001", "Product not found"),
    PRODUCT_OFF_SHELF("3002", "Product off shelf"),
    PRODUCT_CREATE_FAILED("3003", "Product creation failed"),
    PRODUCT_UPDATE_FAILED("3004", "Product update failed"),
    PRODUCT_DELETE_FAILED("3005", "Product deletion failed"),
    PRODUCT_IMAGE_UPLOAD_FAILED("3006", "Product image upload failed"),

    ORDER_NOT_FOUND("4001", "Order not found"),
    ORDER_CREATE_FAILED("4002", "Order creation failed"),
    ORDER_CANCEL_FAILED("4003", "Order cancellation failed"),
    ORDER_STATUS_INVALID("4004", "Order status does not allow this operation"),

    PAYMENT_INSUFFICIENT_BALANCE("5001", "Insufficient balance"),
    PAYMENT_EXCHANGE_FAILED("5002", "Exchange failed"),
    PAYMENT_PRICE_NOT_FOUND("5003", "Price not found"),

    MEDIA_NOT_FOUND("6001", "File not found"),
    MEDIA_UPLOAD_FAILED("6002", "File upload failed"),
    MEDIA_DELETE_FAILED("6003", "File deletion failed"),

    WAREHOUSE_NOT_FOUND("7001", "Warehouse not found"),
    INBOUND_ORDER_NOT_FOUND("7002", "Inbound order not found"),

    SUPPLIER_NOT_FOUND("8001", "Supplier not found"),
    SUPPLIER_CREATE_FAILED("8002", "Supplier creation failed"),
    SUPPLIER_UPDATE_FAILED("8003", "Supplier update failed"),
    SUPPLIER_DELETE_FAILED("8004", "Supplier deletion failed"),
    SUPPLIER_VERIFICATION_FAILED("8005", "Supplier verification failed"),

    PARAM_ERROR("1000", "Parameter error"),
    METHOD_NOT_ALLOWED("1004", "Request method not allowed"),
    NOT_FOUND("1005", "Resource not found"),
    FILE_TOO_LARGE("1006", "File size exceeds limit"),
    FORBIDDEN("1007", "Access forbidden"),

    SYSTEM_ERROR("9999", "System error"),

    // Finance AR/AP
    AR_NOT_FOUND("9001", "Accounts receivable not found"),
    AP_NOT_FOUND("9002", "Accounts payable not found");

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