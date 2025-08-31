package com.metawebthree.media;

import com.alibaba.excel.EasyExcel;
import com.alibaba.excel.context.AnalysisContext;
import com.alibaba.excel.event.AnalysisEventListener;
import com.metawebthree.media.DO.ArtWorkDO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.net.URL;
import java.util.*;
import java.util.function.BiFunction;

@Slf4j
@Service
@RequiredArgsConstructor
public class ExcelService {
    
    private final ArtWorkMapper artworkMapper;
    private static final int BATCH_SIZE = 1000;
    
    // 中文标题到字段名的映射
    private static final Map<String, String> HEADER_MAPPING = new HashMap<>();
    static {
        HEADER_MAPPING.put("作品ID", "id");
        HEADER_MAPPING.put("标题", "title");
        HEADER_MAPPING.put("封面链接", "cover");
        HEADER_MAPPING.put("详情链接", "link");
        HEADER_MAPPING.put("副标题", "subtitle");
        HEADER_MAPPING.put("季数", "season");
        HEADER_MAPPING.put("集数", "episode");
        HEADER_MAPPING.put("类别", "categoryId");
        HEADER_MAPPING.put("标签", "tags");
        HEADER_MAPPING.put("年份标签", "yearTag");
        HEADER_MAPPING.put("演员", "acts");
        HEADER_MAPPING.put("导演", "director");
    }
    
    // 字段处理器接口
    private interface FieldProcessor {
        Object process(String value, ArtWorkDO entity);
    }
    
    // 字段处理器映射
    private static final Map<String, FieldProcessor> FIELD_PROCESSORS = new HashMap<>();
    static {
        // 分类处理器 - 根据名称查找或创建分类
        FIELD_PROCESSORS.put("categoryId", (value, entity) -> {
            // TODO: 实现分类查找/创建逻辑
            return 0; // 默认分类ID
        });
        
        // 标签处理器 - 分割标签并查找/创建
        FIELD_PROCESSORS.put("tags", (value, entity) -> {
            // TODO: 实现标签分割和查找/创建逻辑
            return value; // 暂时返回原始值
        });
        
        // 年份标签处理器 - 智能提取年份
        FIELD_PROCESSORS.put("yearTag", (value, entity) -> {
            if (value != null && !value.isEmpty()) {
                return value;
            }
            // 从标题提取年份
            if (entity.getTitle() != null) {
                String title = entity.getTitle();
                // 匹配标题末尾的4位数字年份
                java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("(\\d{4})$");
                java.util.regex.Matcher matcher = pattern.matcher(title);
                if (matcher.find()) {
                    int year = Integer.parseInt(matcher.group(1));
                    int currentYear = java.time.Year.now().getValue();
                    if (year >= 1969 && year <= currentYear) {
                        return String.valueOf(year);
                    }
                }
            }
            return null;
        });
        
        // 演员处理器
        FIELD_PROCESSORS.put("acts", (value, entity) -> {
            // TODO: 实现演员查找/创建逻辑
            return value;
        });
        
        // 导演处理器
        FIELD_PROCESSORS.put("director", (value, entity) -> {
            // TODO: 实现导演查找/创建逻辑
            return value;
        });
    }
    
    // 字段默认值
    private static final Map<String, Object> DEFAULT_VALUES = Map.of(
        "season", 1,
        "episode", 1,
        "categoryId", 0
    );
    
    public void processExcelData(String excelUrl, int batchSize) {
        try (InputStream inputStream = new URL(excelUrl).openStream()) {
            EasyExcel.read(inputStream, new CustomExcelListener(
                artworkMapper, 
                Math.min(batchSize, BATCH_SIZE),
                this::handleMissingField
            )).sheet().doRead();
            
            log.info("Excel data processing completed successfully");
        } catch (Exception e) {
            log.error("Failed to process Excel data", e);
            throw new RuntimeException("Failed to process Excel data", e);
        }
    }
    
    // 处理缺失字段的回调方法
    private Object handleMissingField(ArtWorkDO data, String fieldName) {
        // 如果有默认值则返回默认值
        if (DEFAULT_VALUES.containsKey(fieldName)) {
            return DEFAULT_VALUES.get(fieldName);
        }
        
        // 根据其他字段计算值
        switch (fieldName) {
            case "yearTag":
                return extractYearFromTitle(data.getTitle());
            default:
                return null;
        }
    }
    
    private String extractYearFromTitle(String title) {
        if (title == null) return null;
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("(\\d{4})$");
        java.util.regex.Matcher matcher = pattern.matcher(title);
        if (matcher.find()) {
            int year = Integer.parseInt(matcher.group(1));
            int currentYear = java.time.Year.now().getValue();
            if (year >= 1969 && year <= currentYear) {
                return String.valueOf(year);
            }
        }
        return null;
    }
    
    // 自定义Excel监听器
    private static class CustomExcelListener extends AnalysisEventListener<Map<Integer, String>> {
        private final ArtWorkMapper mapper;
        private final int batchSize;
        private final BiFunction<ArtWorkDO, String, Object> missingFieldHandler;
        private List<ArtWorkDO> dataList = new ArrayList<>();
        private Map<Integer, String> columnMapping;
        
        public CustomExcelListener(ArtWorkMapper mapper, int batchSize, 
                                 BiFunction<ArtWorkDO, String, Object> missingFieldHandler) {
            this.mapper = mapper;
            this.batchSize = batchSize;
            this.missingFieldHandler = missingFieldHandler;
        }
        
        @Override
        public void invokeHeadMap(Map<Integer, String> headMap, AnalysisContext context) {
            // 解析中文标题，建立列索引到字段名的映射
            columnMapping = new HashMap<>();
            headMap.forEach((index, chineseHeader) -> {
                String fieldName = HEADER_MAPPING.get(chineseHeader);
                if (fieldName != null) {
                    columnMapping.put(index, fieldName);
                }
            });
        }
        
        @Override
        public void invoke(Map<Integer, String> data, AnalysisContext context) {
            ArtWorkDO entity = new ArtWorkDO();
            
            // 根据映射关系设置字段值
            data.forEach((index, value) -> {
                String fieldName = columnMapping.get(index);
                if (fieldName != null && value != null && !value.isEmpty()) {
                    FieldProcessor processor = FIELD_PROCESSORS.get(fieldName);
                    if (processor != null) {
                        setFieldValue(entity, fieldName, processor.process(value, entity));
                    } else {
                        setFieldValue(entity, fieldName, value);
                    }
                }
            });
            
            // 检查并处理缺失的必填字段
            HEADER_MAPPING.values().forEach(fieldName -> {
                try {
                    if (getFieldValue(entity, fieldName) == null) {
                        Object defaultValue = missingFieldHandler.apply(entity, fieldName);
                        if (defaultValue != null) {
                            setFieldValue(entity, fieldName, defaultValue);
                        }
                    }
                } catch (Exception e) {
                    log.warn("Failed to check field {}: {}", fieldName, e.getMessage());
                }
            });
            
            dataList.add(entity);
            if (dataList.size() >= batchSize) {
                mapper.insert(dataList);
                dataList.clear();
            }
        }
        
        @Override
        public void doAfterAllAnalysed(AnalysisContext context) {
            if (!dataList.isEmpty()) {
                mapper.insert(dataList);
            }
        }
        
        private void setFieldValue(ArtWorkDO entity, String fieldName, Object value) {
            try {
                if (value == null) return;
                
                // 根据字段类型设置值
                switch (fieldName) {
                    case "id":
                    case "season":
                    case "episode":
                    case "categoryId":
                        entity.getClass().getMethod("set" + capitalize(fieldName), Integer.class)
                            .invoke(entity, value instanceof Integer ? (Integer)value : Integer.parseInt(value.toString()));
                        break;
                    default:
                        entity.getClass().getMethod("set" + capitalize(fieldName), String.class)
                            .invoke(entity, value.toString());
                }
            } catch (Exception e) {
                log.warn("Failed to set field {}: {}", fieldName, e.getMessage());
            }
        }
        
        private Object getFieldValue(ArtWorkDO entity, String fieldName) {
            try {
                return entity.getClass().getMethod("get" + capitalize(fieldName)).invoke(entity);
            } catch (Exception e) {
                return null;
            }
        }
        
        private String capitalize(String str) {
            return str.substring(0, 1).toUpperCase() + str.substring(1);
        }
    }
}
