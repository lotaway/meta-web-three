package com.metawebthree.user.interfaces.web;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.util.Arrays;

import org.junit.jupiter.api.Test;
import org.springframework.web.bind.annotation.PostMapping;

class UserControllerAnnotationTest {

    @Test
    void signIn_shouldBePostOnly() throws Exception {
        Method method = UserController.class.getDeclaredMethod(
                "signIn",
                Long.class,
                String.class,
                String.class,
                Integer.class);

        PostMapping postMapping = method.getAnnotation(PostMapping.class);
        assertNotNull(postMapping);
        assertTrue(Arrays.asList(postMapping.value()).contains("/signIn"));
    }
}
