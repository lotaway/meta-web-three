package com.metawebthree.product.exception;

public class ProductDomainException extends RuntimeException {

    private final ProductErrorCode errorCode;

    public ProductDomainException(ProductErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public ProductDomainException(ProductErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    public ProductErrorCode getErrorCode() {
        return errorCode;
    }
}
