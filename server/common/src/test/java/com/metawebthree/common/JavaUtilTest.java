package com.metawebthree.common;

import com.metawebthree.common.utils.JavaUtil;
import org.junit.Assert;
import org.junit.Test;

import java.util.List;

public class JavaUtilTest {

    private final JavaUtil supportUtil = new JavaUtil();

    @Test
    public void getInsertPosTest() {
        int shouldBeThree = supportUtil.getInsertPos(List.of(0, 1, 236, 5567, 12546), 315);
        Assert.assertEquals(3, shouldBeThree);

        int shouldBeFirstOne = supportUtil.getInsertPos(List.of(), 4);
        Assert.assertEquals(0, shouldBeFirstOne);

        int shouldBeLastOne = supportUtil.getInsertPos(List.of(0, 1, 236, 5567, 12546), 12546);
        Assert.assertEquals(4, shouldBeLastOne);

        int shouldBeZero = supportUtil.getInsertPos(List.of(44), 43);
        Assert.assertEquals(0, shouldBeZero);

        int shouldBeNewLast = supportUtil.getInsertPos(List.of(44), 45);
        Assert.assertEquals(1, shouldBeNewLast);
    }

}
