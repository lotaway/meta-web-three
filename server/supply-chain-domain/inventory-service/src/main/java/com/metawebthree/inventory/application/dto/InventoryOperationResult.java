package com.metawebthree.inventory.application.dto;

import lombok.Data;

@Data
public class InventoryOperationResult {
    private boolean success;
    private String message;
    private String bizId;
    private Integer quantity;

    public static InventoryOperationResult success(String bizId, Integer quantity) {
        InventoryOperationResult result = new InventoryOperationResult();
        result.setSuccess(true);
        result.setMessage("Operation successful");
        result.setBizId(bizId);
        result.setQuantity(quantity);
        return result;
    }

    public static InventoryOperationResult fail(String message) {
        InventoryOperationResult result = new InventoryOperationResult();
        result.setSuccess(false);
        result.setMessage(message);
        return result;
    }
}