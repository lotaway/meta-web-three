package com.metawebthree.finance.domain.exception;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;

public class ArNotFoundException extends BusinessException {

    public ArNotFoundException(Long id) {
        super(ResponseStatus.AR_NOT_FOUND, "AR ID [" + id + "] not found");
    }

    public ArNotFoundException(String message) {
        super(ResponseStatus.AR_NOT_FOUND, message);
    }
}