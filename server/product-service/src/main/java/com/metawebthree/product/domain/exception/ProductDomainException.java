package com.metawebthree.product.exception;

/**
 * 产品领域异常
 * 用于封装产品领域内的错误，提供可分类、可定位、可追责的异常信息
 */
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
