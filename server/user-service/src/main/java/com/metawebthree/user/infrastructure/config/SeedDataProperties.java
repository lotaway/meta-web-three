package com.metawebthree.user.infrastructure.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "seed-data")
public class SeedDataProperties {

    private long superAdminRoleId = 3001L;
    private long defaultAdminId = 1L;
    private ResourceRange resource = new ResourceRange();
    private MenuRange menu = new MenuRange();

    public long getSuperAdminRoleId() { return superAdminRoleId; }
    public void setSuperAdminRoleId(long superAdminRoleId) { this.superAdminRoleId = superAdminRoleId; }
    public long getDefaultAdminId() { return defaultAdminId; }
    public void setDefaultAdminId(long defaultAdminId) { this.defaultAdminId = defaultAdminId; }
    public ResourceRange getResource() { return resource; }
    public void setResource(ResourceRange resource) { this.resource = resource; }
    public MenuRange getMenu() { return menu; }
    public void setMenu(MenuRange menu) { this.menu = menu; }

    public static class ResourceRange {
        private long startId = 101L;
        private long endId = 410L;

        public long getStartId() { return startId; }
        public void setStartId(long startId) { this.startId = startId; }
        public long getEndId() { return endId; }
        public void setEndId(long endId) { this.endId = endId; }
    }

    public static class MenuRange {
        private long startId = 2001L;
        private long endId = 2042L;

        public long getStartId() { return startId; }
        public void setStartId(long startId) { this.startId = startId; }
        public long getEndId() { return endId; }
        public void setEndId(long endId) { this.endId = endId; }
    }
}
