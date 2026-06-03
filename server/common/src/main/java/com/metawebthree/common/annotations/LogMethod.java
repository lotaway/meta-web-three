package com.metawebthree.common.annotations;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation for logging method execution and auditing operations
 * @param value Operation description for audit trail (e.g., "Create User", "Update Order")
 * @param param Whether to log method parameters
 * @param result Whether to log method return value
 * @param timeCost Whether to log method execution time
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface LogMethod {
    String value() default "";  // Operation description for audit
    boolean param() default true;
    boolean result() default true;
    boolean timeCost() default false;
}