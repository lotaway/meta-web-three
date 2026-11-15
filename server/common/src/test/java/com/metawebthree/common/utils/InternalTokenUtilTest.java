package com.metawebthree.common.utils;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class InternalTokenUtilTest {

    @Test
    void generatedTokenIsAccepted() {
        InternalTokenUtil util = new InternalTokenUtil("shared-secret", 5);
        assertTrue(util.validate(util.generate()));
    }

    @Test
    void tokenSignedWithAnotherSecretIsRejected() {
        String token = new InternalTokenUtil("shared-secret", 5).generate();
        assertFalse(new InternalTokenUtil("other-secret", 5).validate(token));
    }

    @Test
    void tamperedSignatureIsRejected() {
        InternalTokenUtil util = new InternalTokenUtil("shared-secret", 5);
        String token = util.generate();
        assertFalse(util.validate(token.substring(0, token.length() - 2) + "xx"));
    }

    @Test
    void malformedTokensAreRejected() {
        InternalTokenUtil util = new InternalTokenUtil("shared-secret", 5);
        assertFalse(util.validate(null));
        assertFalse(util.validate(""));
        assertFalse(util.validate("abc"));
        assertFalse(util.validate("123:"));
        assertFalse(util.validate(":signature"));
        assertFalse(util.validate("not-a-number:signature"));
    }

    @Test
    void tokenOutsideTheValidityWindowIsRejected() {
        // A negative window makes even a freshly generated token stale.
        InternalTokenUtil util = new InternalTokenUtil("shared-secret", -1);
        assertFalse(util.validate(util.generate()));
    }
}
