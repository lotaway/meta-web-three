package com.metawebthree.finance.domain.exception;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;

public class ApNotFoundException extends BusinessException {

    public ApNotFoundException(Long id) {
        super(ResponseStatus.AP_NOT_FOUND, "AP ID [" + id + "] not found");
    }

    public ApNotFoundException(String message) {
        super(ResponseStatus.AP_NOT_FOUND, message);
    }
}