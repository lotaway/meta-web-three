package com.metawebthree.inventory.application.dto;

import lombok.Data;

@Data
public class InventoryOperationResult {
    private boolean success;
    private String message;
    private String bizId;
    private Integer quantity;
    /** True when the operation was a no-op because it had already been applied. */
    private boolean duplicated;

    public static InventoryOperationResult success(String bizId, Integer quantity) {
        InventoryOperationResult result = new InventoryOperationResult();
        result.setSuccess(true);
        result.setMessage("Operation successful");
        result.setBizId(bizId);
        result.setQuantity(quantity);
        return result;
    }

    public static InventoryOperationResult successDuplicated(String bizId, Integer quantity) {
        InventoryOperationResult result = success(bizId, quantity);
        result.setDuplicated(true);
        result.setMessage("Operation already applied");
        return result;
    }

    public static InventoryOperationResult fail(String message) {
        InventoryOperationResult result = new InventoryOperationResult();
        result.setSuccess(false);
        result.setMessage(message);
        return result;
    }
}