package com.metawebthree.common;

import com.metawebthree.common.utils.JavaUtil;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

public class JavaUtilTest {

    private final JavaUtil supportUtil = new JavaUtil();

    @Test
    public void getInsertPosTest() {
        int shouldBeThree = supportUtil.getInsertPos(List.of(0, 1, 236, 5567, 12546), 315);
        Assertions.assertEquals(3, shouldBeThree);

        int shouldBeFirstOne = supportUtil.getInsertPos(List.of(), 4);
        Assertions.assertEquals(0, shouldBeFirstOne);

        int shouldBeLastOne = supportUtil.getInsertPos(List.of(0, 1, 236, 5567, 12546), 12546);
        Assertions.assertEquals(4, shouldBeLastOne);

        int shouldBeZero = supportUtil.getInsertPos(List.of(44), 43);
        Assertions.assertEquals(0, shouldBeZero);

        int shouldBeNewLast = supportUtil.getInsertPos(List.of(44), 45);
        Assertions.assertEquals(1, shouldBeNewLast);
    }

}
