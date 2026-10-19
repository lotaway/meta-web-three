package com.metawebthree.wallet.domain.model;

/**
 * Wallet细粒度权限枚举。
 * 每个钱包操作对应一个权限，确保最小权限原则。
 */
public enum WalletPermission {

    WALLET_CREATE("wallet:create", "创建钱包"),
    WALLET_READ("wallet:read", "查询钱包"),
    WALLET_DEPOSIT("wallet:deposit", "充值"),
    WALLET_WITHDRAW("wallet:withdraw", "提现"),
    WALLET_ADMIN("wallet:admin", "管理员全权");

    private final String code;
    private final String description;

    WalletPermission(String code, String description) {
        this.code = code;
        this.description = description;
    }

    public String getCode() {
        return code;
    }

    public String getDescription() {
        return description;
    }
}