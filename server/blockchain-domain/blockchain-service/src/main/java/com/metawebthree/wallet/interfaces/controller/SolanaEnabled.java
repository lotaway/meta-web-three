package com.metawebthree.wallet.interfaces.controller;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;

/**
 * Gates Solana/virtual-asset REST endpoints behind a single opt-in switch.
 * Crypto features are optional: the normal (fiat) mall must work with them off.
 */
@Target({ ElementType.TYPE, ElementType.METHOD })
@Retention(RetentionPolicy.RUNTIME)
@Documented
@ConditionalOnProperty(prefix = "blockchain.solana", name = "enabled", havingValue = "true")
public @interface SolanaEnabled {
}
