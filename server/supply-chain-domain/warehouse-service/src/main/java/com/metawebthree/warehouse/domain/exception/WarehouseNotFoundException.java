package com.metawebthree.warehouse.domain.exception;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;

/**
 * 仓库不存在异常
 */
public class WarehouseNotFoundException extends BusinessException {

    public WarehouseNotFoundException(Long id) {
        super(ResponseStatus.WAREHOUSE_NOT_FOUND, "仓库ID[" + id + "]不存在");
    }

    public WarehouseNotFoundException(String message) {
        super(ResponseStatus.WAREHOUSE_NOT_FOUND, message);
    }
}