package com.metawebthree.common.utils;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class AlgorithmUtils {

    public static int[] testData() throws IOException {
        File file = new File("./AlgorithmData.txt");
        FileReader fileReader = new FileReader(file);
        char[] buffer = new char[(int) file.length()];
        fileReader.read(buffer);
        String[] strNums = new String(buffer).split(",");
        int[] data = new int[strNums.length];
        int dataIndex = 0;
        for (String strNum : strNums) {
            data[dataIndex++] = Integer.parseInt(strNum);
        }
        fileReader.close();
        return data;
    }

    public int maxArea(int[] heights) {
        if (heights.length == 0)
            return -1;
        int max = 0;
        for (int i = 0, j = heights.length - 1; i < j; ) {
            max = Math.max(max, (j - i) * (heights[i] < heights[j] ? heights[i++] : heights[j--]));
        }
        return max;
    }

    public void maxAreaTest() throws IOException {
        long startTime = System.currentTimeMillis();
        int result = maxArea(testData());
        long endTime = System.currentTimeMillis();
        System.out.println(result + ", time:" + (endTime - startTime) / 1000);
    }

    /*public static void main(String[] args) throws IOException {
        new AlgorithmUtils().maxAreaTest();
    }*/
}
