package com.metawebthree.common.annotations;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface LogMethod {
    boolean param() default true;
    boolean result() default true;
    boolean timeCost() default false;
}