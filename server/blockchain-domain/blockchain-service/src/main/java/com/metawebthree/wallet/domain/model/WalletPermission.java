package com.metawebthree.wallet.domain.model;

/**
 * 钱包细粒度权限常量。
 *
 * <p>与 {@code ERPPermissions} 保持一致的常量类风格：权限码为编译期常量，
 * 可直接用于 {@code @RequirePermission} 注解等需要常量表达式的位置。</p>
 */
public final class WalletPermission {

    private WalletPermission() {}

    public static final String WALLET_CREATE  = "wallet:create";
    public static final String WALLET_READ    = "wallet:read";
    public static final String WALLET_DEPOSIT = "wallet:deposit";
    public static final String WALLET_WITHDRAW = "wallet:withdraw";
    public static final String WALLET_ADMIN   = "wallet:admin";
}
