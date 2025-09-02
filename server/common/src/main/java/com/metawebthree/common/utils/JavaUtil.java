package com.metawebthree.common.utils;

import java.io.File;
import java.net.InetAddress;
import java.util.List;

public class JavaUtil {

    public int getInsertPos(List<Integer> arr, int k) {
        if (arr.size() == 0)
            return 0;
        if (arr.size() == 1)
            return arr.get(0) >= k ? 0 : 1;
        int left = 0, right = arr.size() - 1, index = arr.size() / 2;
        while (left < right - 1) {
            if (arr.get(index) == k) {
                return index;
            }
            if (arr.get(index) > k) {
                right = index;
            } else {
                left = index;
            }
            index = (left + right + 1) / 2;
        }
        if (left != right)
            return arr.get(index) > arr.get(left) && arr.get(index) <= arr.get(right) ? left + 1 : left;
        return index;
    }

    public String getServiceName() {
        try {
            String jarPath = new File(getClass().getProtectionDomain()
                    .getCodeSource().getLocation().toURI()).getName();
            return jarPath.replaceAll("[^a-zA-Z0-9]", "-");
        } catch (Exception e) {
            return "unknown-service-name-" + System.currentTimeMillis();
        }
    }

    public String getInstanceId() {
        try {
            return InetAddress.getLocalHost().getHostName();
                    // + "-" + ProcessHandle.current().pid()
                    // + "-" + System.currentTimeMillis();
        } catch (Exception e) {
            return "unknown-instance-id-" + System.currentTimeMillis();
        }
    }

}