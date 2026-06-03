package com.meta.common.audit;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface Audit {

    String operationType() default "";

    String resourceType() default "";

    String resourceIdParam() default "";

    String description() default "";

    boolean recordParams() default true;

    boolean recordResponse() default false;

    boolean ignoreSensitive() default true;
}
