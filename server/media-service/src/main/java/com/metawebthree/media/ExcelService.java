package com.metawebthree.media;

import com.alibaba.excel.EasyExcel;
import com.alibaba.excel.context.AnalysisContext;
import com.alibaba.excel.event.AnalysisEventListener;
import com.github.yulichang.query.MPJLambdaQueryWrapper;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.media.DO.ArtWorkCategoryDO;
import com.metawebthree.media.DO.ArtWorkDO;
import com.metawebthree.media.DO.PeopleDO;
import com.metawebthree.media.DO.PeopleTypeDO;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.springframework.stereotype.Service;
import org.web3j.tuples.generated.Tuple2;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.URI;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class ExcelService {

    private final ArtWorkMapper artworkMapper;
    private final ArtWorkCategoryMapper artworkCategoryMapper;
    private final PeopleMapper peopleMapper;
    private final PeopleTypeMapper peopleTypeMapper;
    private static final int BATCH_SIZE = 1000;
    private static final int PROCESSING_THREADS = 4;

    private final ExecutorService processingExecutor = Executors.newFixedThreadPool(PROCESSING_THREADS);
    private final ExecutorService insertionExecutor = Executors.newSingleThreadExecutor();
    private final BlockingQueue<List<ArtWorkDO>> processingQueue = new LinkedBlockingQueue<>();
    private final Map<String, String> HEADER_MAPPING = new HashMap<>();

    private interface FieldProcessor {
        Tuple2<String, String> process(String value, ArtWorkDO entity);
    }

    private final Map<String, FieldProcessor> FIELD_PROCESSORS = new HashMap<>();

    @PostConstruct
    private void initialize() {
        HEADER_MAPPING.put("作品ID", "id");
        HEADER_MAPPING.put("标题", "title");
        HEADER_MAPPING.put("封面链接", "cover");
        HEADER_MAPPING.put("详情链接", "link");
        HEADER_MAPPING.put("副标题", "subtitle");
        HEADER_MAPPING.put("季数", "season");
        HEADER_MAPPING.put("集数", "episode");
        HEADER_MAPPING.put("类别", "categoryName");
        HEADER_MAPPING.put("标签", "tags");
        HEADER_MAPPING.put("年份标签", "yearTag");
        HEADER_MAPPING.put("演员", "acts");
        HEADER_MAPPING.put("导演", "director");

        FIELD_PROCESSORS.put("categoryName", (categoryName, entity) -> {
            var wrapper = new MPJLambdaQueryWrapper<ArtWorkCategoryDO>();
            wrapper.eq(ArtWorkCategoryDO::getName, categoryName);
            ArtWorkCategoryDO result = artworkCategoryMapper.selectOne(wrapper);
            if (result != null) {
                return new Tuple2<String, String>("categoryId", result.getId().toString());
            }
            var categoryDO = ArtWorkCategoryDO.builder().name(categoryName).build();
            artworkCategoryMapper.insert(categoryDO);
            return new Tuple2<String, String>("categoryId", categoryDO.getId().toString());
        });

        FIELD_PROCESSORS.put("tags", (value, entity) -> {
            // TODO: 实现标签分割和查找/创建逻辑
            return new Tuple2<String, String>("tags", value);
        });

        FIELD_PROCESSORS.put("yearTag", (value, entity) -> {
            if (value != null && !value.isEmpty()) {
                return new Tuple2<String, String>("yearTag", value);
            }
            Optional<Integer> result = getYearFromTitle(entity.getTitle());
            return new Tuple2<String, String>("yearTag", result.isEmpty() ? "" : result.get().toString());
        });

        FIELD_PROCESSORS.put("acts", (actorNames, entity) -> {
            var actorNameList = List.<String>of(actorNames.split(","));
            var actorIdList = new ArrayList<Integer>();
            actorNameList.forEach((String actorName) -> {
                var wrapper = new MPJLambdaWrapper<PeopleDO>();
                wrapper.select(PeopleDO::getId).eq(PeopleDO::getName, actorName).leftJoin(PeopleTypeDO.class,
                        PeopleTypeDO::getId, PeopleDO::getTypes);
                List<PeopleDO> result = peopleMapper.selectJoinList(wrapper);
                PeopleDO peopleDO;
                if (result == null || result.isEmpty()) {
                    List<PeopleTypeDO> typeDOs = peopleTypeMapper.selectList(new MPJLambdaWrapper<PeopleTypeDO>()
                            .select(PeopleTypeDO::getId).eq(PeopleTypeDO::getType, actorName));
                    peopleDO = PeopleDO.builder().name(actorName).types(new Short[] { typeDOs.get(0).getId() }).build();
                    peopleMapper.insert(peopleDO);
                } else {
                    peopleDO = result.get(0);
                }
                actorIdList.add(peopleDO.getId());
            });
            if (!actorIdList.isEmpty()) {
                return new Tuple2<String, String>("acts",
                        actorIdList.stream().map(String::valueOf).collect(Collectors.joining(",")));
            }
            return new Tuple2<String, String>("acts", "");
        });

        FIELD_PROCESSORS.put("director", (value, entity) -> {
            // TODO: 实现导演查找/创建逻辑
            return value;
        });
    }

    public static Optional<Integer> getYearFromTitle(String str) {
        var BASIC_YEAR = 1950;
        if (str != null) {
            var pattern = java.util.regex.Pattern.compile("(\\d{4})$");
            var matcher = pattern.matcher(str);
            if (matcher.find()) {
                var year = Integer.parseInt(matcher.group(1));
                var currentYear = java.time.Year.now().getValue();
                if (year >= BASIC_YEAR && year <= currentYear) {
                    return Optional.of(year);
                }
            }
        }
        return Optional.empty();
    }

    private static final Map<String, Object> DEFAULT_VALUES = Map.of(
            "season", 1,
            "episode", 1);

    public void processExcelData(String excelUrl, int batchSize) {
        try (InputStream inputStream = new URI(excelUrl).toURL().openStream()) {
            EasyExcel.read(inputStream, new CustomExcelListener(
                    artworkMapper,
                    Math.min(batchSize, BATCH_SIZE),
                    this::handleMissingField)).sheet().doRead();

            log.info("Excel data processing completed successfully");
        } catch (Exception e) {
            log.error("Failed to process Excel data", e);
            throw new RuntimeException("Failed to process Excel data", e);
        }
    }

    private Object handleMissingField(ArtWorkDO data, String fieldName) {
        if (DEFAULT_VALUES.containsKey(fieldName)) {
            return DEFAULT_VALUES.get(fieldName);
        }
        switch (fieldName) {
            case "yearTag":
                return extractYearFromTitle(data.getTitle());
            default:
                return null;
        }
    }

    private String extractYearFromTitle(String title) {
        if (title == null)
            return null;
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

    private class CustomExcelListener extends AnalysisEventListener<Map<Integer, String>> {
        private final ArtWorkMapper mapper;
        private final int batchSize;
        private final BiFunction<ArtWorkDO, String, Object> missingFieldHandler;
        private final List<Map<Integer, String>> rawDataBuffer = new ArrayList<>();
        private Map<Integer, String> columnMapping;
        private final CountDownLatch completionLatch = new CountDownLatch(1);

        public CustomExcelListener(ArtWorkMapper mapper, int batchSize,
                BiFunction<ArtWorkDO, String, Object> missingFieldHandler) {
            this.mapper = mapper;
            this.batchSize = batchSize;
            this.missingFieldHandler = missingFieldHandler;

            // Start insertion worker thread
            insertionExecutor.execute(() -> {
                try {
                    while (!Thread.currentThread().isInterrupted()) {
                        List<ArtWorkDO> batch = processingQueue.take();
                        if (!batch.isEmpty()) {
                            mapper.insert(batch);
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        @Override
        public void invokeHeadMap(Map<Integer, String> headMap, AnalysisContext context) {
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
            if (columnMapping == null) {
                String msg = "columnMapping for head is not initialized, maybe invokeHeadMap not invoked";
                log.error(msg, context);
                throw new IllegalStateException(msg);
            }

            rawDataBuffer.add(data);
            if (rawDataBuffer.size() >= batchSize) {
                processBatchInBackground(new ArrayList<>(rawDataBuffer));
                rawDataBuffer.clear();
            }
        }

        private void processBatchInBackground(List<Map<Integer, String>> batchData) {
            processingExecutor.submit(() -> {
                List<ArtWorkDO> processedBatch = new ArrayList<>();
                for (Map<Integer, String> data : batchData) {
                    ArtWorkDO entity = new ArtWorkDO();

                    data.forEach((index, value) -> {
                        String fieldName = columnMapping.get(index);
                        if (fieldName != null && value != null && !value.isEmpty()) {
                            FieldProcessor processor = FIELD_PROCESSORS.get(fieldName);
                            if (processor != null) {
                                var result = processor.process(value, entity);
                                fieldName = result.component1();
                                value = result.component2();
                            }
                            setFieldValue(entity, fieldName, value);
                        }
                    });

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

                    processedBatch.add(entity);
                }

                try {
                    processingQueue.put(processedBatch);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    log.error("Interrupted while adding batch to queue", e);
                }
            });
        }

        @Override
        public void doAfterAllAnalysed(AnalysisContext context) {
            if (!rawDataBuffer.isEmpty()) {
                processBatchInBackground(new ArrayList<>(rawDataBuffer));
            }

            insertionExecutor.shutdown();
            processingExecutor.shutdown();

            try {
                if (!insertionExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                    insertionExecutor.shutdownNow();
                }
                if (!processingExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                    processingExecutor.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.error("Interrupted while waiting for executor shutdown", e);
            }
        }

        private void setFieldValue(ArtWorkDO entity, String fieldName, Object value) {
            try {
                if (value == null)
                    return;
                switch (fieldName) {
                    case "id":
                    case "season":
                    case "episode":
                    case "categoryId":
                        entity.getClass().getMethod("set" + capitalize(fieldName), Integer.class)
                                .invoke(entity, value instanceof Integer integer ? integer
                                        : Integer.parseInt(value.toString()));
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

    @PreDestroy
    public void onDestroy() {
        processingExecutor.shutdown();
        insertionExecutor.shutdown();
        try {
            if (!processingExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                processingExecutor.shutdownNow();
            }
            if (!insertionExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                insertionExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            processingExecutor.shutdownNow();
            insertionExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    public byte[] generateTemplate() {
        try (ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
            List<String> headers = new ArrayList<>(HEADER_MAPPING.keySet());
            List<Object> exampleRow1 = Arrays.asList(
                    "",
                    "示例作品2023",
                    "https://example.com/cover1.jpg",
                    "https://example.com/detail1",
                    "示例副标题",
                    1,
                    10,
                    "类别如Movie",
                    "标签1,标签2",
                    "2023",
                    "演员1,演员2",
                    "导演1");
            List<Object> exampleRow2 = Arrays.asList(
                    "",
                    "超人联盟",
                    "https://youku.com/gis/superman_legend.jpg",
                    "https://youku.com/video/superman_legend",
                    "假期大作战",
                    1,
                    1,
                    "Movie",
                    "喜剧，超能力，动画",
                    "2022",
                    "凯奇·古丽",
                    "雷蒙·利特波");
            EasyExcel.write(outputStream)
                    .head(Collections.singletonList(headers))
                    .sheet("template")
                    .doWrite(Arrays.asList(exampleRow1, exampleRow2));

            return outputStream.toByteArray();
        } catch (Exception e) {
            log.error("Failed to generate Excel template", e);
            throw new RuntimeException("Failed to generate Excel template", e);
        }
    }
}
