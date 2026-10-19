package com.metawebthree.mes.domain.service;

/**
 * Exception thrown when a split operation is not yet implemented.
 * <p>
 * Used to clearly mark split strategies that are planned but not yet implemented.
 * API consumers and documentation should treat this as "planned feature, not yet available"
 * rather than a system error.
 */
public class SplitNotSupportedException extends RuntimeException {

    public SplitNotSupportedException(String reason) {
        super(reason);
    }

    public SplitNotSupportedException(String reason, Throwable cause) {
        super(reason, cause);
    }
}