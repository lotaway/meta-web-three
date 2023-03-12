package com.init;

import java.util.Scanner;
import java.util.Random;

public class InitScanner {

    private String welcomeTitle;

    public InitScanner() {
        System.out.println("Running InitScanner");
    }

    public InitScanner(String _welcomeTitle) {
        this.welcomeTitle = _welcomeTitle;
    }

    public static void main(String[] args) {
        System.out.println("Into init");

        //  基础类型

        //  byte,short,char相加时自动转为int
        byte b = 10;    //  1个字节，等于0000 0010
        short s = 20;   //  2个字节，等于0000 0000 0000 0100
        char c = 'a'; //    'A'=65,'a'=97
        int i1 = b + s + c; //  4个字节，等于0000 0000 0000 0000 0000 0000 0000 0000

        //  int,long, float, double相加时隐性地向更大范围的类型转换
        int i2 = 10;
        long n = 100L;
        double d = 20.5;
        double d2 = i2 + n + d;

        //  引用类型：类、接口、数组、字符串、null
        //  直接赋值会在栈上分配内存，在串池创建该常量字符串，并让变量指向该常量字符串在串池的地址
        String title = "Into init";
        //  重新赋值不会修改常量字符串，而是重新在串池创建一个常量字符串，之后让变量指向这个新地址
        title = "Done";
        //  赋值时，会先在串池中寻找是否有相同值的字符串，若有的话就直接将变量指向这个字符串的串池地址，而不会创建
        title = "Into init";
        //  遇到字符串的相加前面的值会自动变成字符串
        String title2 = title + d2;
        //  完成数值相加后才变成字符串
        String title3 = i2 + n + title;
        //  字符和字符串相加会按照字符串拼接处理
        String title4 = c + title2;

        //  强制转换，可能丢失精度
        b = (byte) i1;
        i2 = (int) (i2 + n + d);

        boolean isMatch = b == i2;

        //  数组静态定义方式，定义的同时赋值。以下两种方式完全相同，初始化列表会隐形使用new int[]进行类型转换
        int[] intArr1 = {1, 2, 3};
        int intArr2[] = {1, 2, 3};
        if (intArr1[0] == intArr2[0]) {
            int len = intArr1.length;
        }
        //  数组动态定义，没有在定义的时候赋值
        String[] strArr3 = new String[3];
        strArr3[0] = "wayluk ";
        strArr3[1] = "is 30 ";
        strArr3[2] = "or not ";

        //  随机数，需要引入包java.util.Random
        Random r = new Random();
        for (int j = 0, l = intArr1.length; j < l; j++) {
            if (intArr1[j] > r.nextInt(10)) {
                strArr3[j] = j + ":" + intArr1[j];
            }
        }

        Scanner sc = new Scanner(System.in);
        //  打印可以通过输入sout快捷键自动输出以下方法
        System.out.println("Please input a number:");
        int first_val = sc.nextInt();
        System.out.println("Please input another number");
        int second_val = sc.nextInt();
        System.out.println("Multiply result: " + (first_val * second_val));


        int ret1 = add(1, 2);
        double ret2 = add(1.1, 2.1);
    }

    public static int add(int num1, int num2) {
        return num1 + num2;
    }

    //  重载方法，同名方法不同形参列表
    public static double add(double num1, double num2) {
        return num1 + num2;
    }

    //  使用成员变量可以直接忽略this
    public String getWelcomeTitle() {
        return welcomeTitle;
    }

    public void setWelcomeTitle(String welcomeTitle) {
        this.welcomeTitle = welcomeTitle;
    }

    public void initStringHandle() {
        char[] cArr = {'h', 'l'};
        //  字符串堆分配，主要用于从byte、char类型数组转换过来
        String title5 = new String(cArr);
        String title6 = new String(cArr);
        if (title5 == title6) {
            System.out.println("it's equals in string pool address");
        }
        if (title5.equals(title6)) {
            System.out.println("it's equals in string pool address or value");
        }
        if (title5.equalsIgnoreCase(title6)) {
            System.out.println("it's equals in string pool address or value ignore upper and lower case");
        }
        //  命令行输入获取到的字符串是通过new创建的
        Scanner sc = new Scanner(System.in);
        String title7 = sc.next();
        if (title7 == title5)
            System.out.println("Not the same address");

        //  循环读取单个字符
        for (int i = 0, l = title7.length(); i < l; i++) {
            char c = title7.charAt(i);
            if (c >= 'a' && c <= 'z') {
                System.out.println("这是小写字母：" + c);
            } else if (c >= 'A' && c <= 'Z') {
                System.out.println("这是大写字母：" + c);
            } else if (c >= '0' && c <= '9') {
                System.out.println("这是数字：" + c);
            } else {
                System.out.println("啥也不是：" + c);
            }
        }
    }
}