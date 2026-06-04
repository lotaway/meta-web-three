package com.metawebthree.common.utils;

import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.exception.BusinessException;

public class ValidationUtils {

    public static long parseLong(String value, String fieldName) {
        if (value == null || value.isBlank())
            throw new BusinessException(ResponseStatus.PARAM_TYPE_ERROR, fieldName + " must be a valid long");
        try { return Long.parseLong(value.trim()); }
        catch (NumberFormatException e) { throw new BusinessException(ResponseStatus.PARAM_TYPE_ERROR, fieldName + " must be a valid long"); }
    }

    public static long parseLongSafe(Object value, String fieldName) {
        if (value == null)
            throw new BusinessException(ResponseStatus.PARAM_TYPE_ERROR, fieldName + " must not be null");
        if (value instanceof Number num) return num.longValue();
        if (value instanceof String str) return parseLong(str, fieldName);
        throw new BusinessException(ResponseStatus.PARAM_TYPE_ERROR, fieldName + " must be a number or numeric string");
    }

    public static int parseInt(Object value, String fieldName) {
        if (value == null)
            throw new BusinessException(ResponseStatus.PARAM_TYPE_ERROR, fieldName + " must not be null");
        if (value instanceof Number num) return num.intValue();
        if (value instanceof String str) {
            try { return Integer.parseInt(str.trim()); }
            catch (NumberFormatException e) { throw new BusinessException(ResponseStatus.PARAM_TYPE_ERROR, fieldName + " must be a valid integer"); }
        }
        throw new BusinessException(ResponseStatus.PARAM_TYPE_ERROR, fieldName + " must be a number or numeric string");
    }

    public static String requireNonBlank(String value, String fieldName) {
        if (value == null || value.isBlank())
            throw new BusinessException(ResponseStatus.PARAM_MISSING_ERROR, fieldName + " must not be blank");
        return value;
    }

    public static <T extends Enum<T>> T validateEnum(Class<T> enumClass, String value, String fieldName) {
        if (value == null || value.isBlank())
            throw new BusinessException(ResponseStatus.PARAM_TYPE_ERROR, fieldName + " must not be null or empty");
        try { return Enum.valueOf(enumClass, value.toUpperCase()); }
        catch (IllegalArgumentException e) { throw new BusinessException(ResponseStatus.PARAM_TYPE_ERROR, fieldName + " must be a valid " + enumClass.getSimpleName()); }
    }

    public static int safeIntFromLong(long value, String fieldName) {
        if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE)
            throw new BusinessException(ResponseStatus.PARAM_TYPE_ERROR, fieldName + " value " + value + " is out of integer range");
        return (int) value;
    }
}
