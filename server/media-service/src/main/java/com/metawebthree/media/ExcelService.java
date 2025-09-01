package com.metawebthree.media;

import com.alibaba.excel.EasyExcel;
import com.alibaba.excel.context.AnalysisContext;
import com.alibaba.excel.event.AnalysisEventListener;
import com.baomidou.mybatisplus.core.metadata.TableInfo;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.yulichang.query.MPJLambdaQueryWrapper;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.media.BO.ExcelTemplateBO;
import com.metawebthree.media.DO.ArtWorkCategoryDO;
import com.metawebthree.media.DO.ArtWorkDO;
import com.metawebthree.media.DO.ArtWorkTagDO;
import com.metawebthree.media.DO.PeopleDO;
import com.metawebthree.media.DO.PeopleTypeDO;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.web3j.tuples.generated.Tuple2;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.URI;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiFunction;
import java.util.function.Predicate;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class ExcelService {

    private final StringRedisTemplate redisTemplate;
    private static final String IMPORT_EXCEL_QUEUE_KEY = "excel-processing-queue";
    private static final String FAILED_DECODE_IMPORT_EXCEL_QUEUE_KEY = "failed-decode-excel-processing-queue";
    private static final String FAILED_ENCODE_IMPORT_EXCEL_QUEUE_KEY = "failed-encode-excel-processing-queue";
    private final ArtWorkMapper artworkMapper;
    private final ArtWorkCategoryMapper artworkCategoryMapper;
    private final PeopleMapper peopleMapper;
    private final PeopleTypeMapper peopleTypeMapper;
    private final ArtWorkTagMapper artworkTagMapper;
    private static final int MIN_BATCH_SIZE = 1;
    private static final int MAX_BATCH_SIZE = 1000;
    private static final int PROCESSING_THREADS = 4;

    private final ExecutorService processingExecutor = Executors.newFixedThreadPool(PROCESSING_THREADS);
    private final ExecutorService insertionExecutor = Executors.newSingleThreadExecutor();

    private interface FieldSetter<T> {
        void setField(String value, T entity);
    }

    private final Map<String, FieldSetter<ExcelTemplateBO>> FIELD_SETTERS = new HashMap<>();

    @PostConstruct
    private void initialize() {
        FIELD_SETTERS.put("作品ID", (String value, ExcelTemplateBO entity) -> entity.setId(value));
        FIELD_SETTERS.put("标题", (String value, ExcelTemplateBO entity) -> entity.setTitle(value));
        FIELD_SETTERS.put("封面链接", (String value, ExcelTemplateBO entity) -> entity.setCover(value));
        FIELD_SETTERS.put("详情链接", (String value, ExcelTemplateBO entity) -> entity.setLink(value));
        FIELD_SETTERS.put("副标题", (String value, ExcelTemplateBO entity) -> entity.setSubtitle(value));
        FIELD_SETTERS.put("季数", (String value, ExcelTemplateBO entity) -> entity.setSeason(value));
        FIELD_SETTERS.put("集数", (String value, ExcelTemplateBO entity) -> entity.setEpisode(value));
        FIELD_SETTERS.put("类别", (String value, ExcelTemplateBO entity) -> entity.setCategoryName(value));
        FIELD_SETTERS.put("标签", (String value, ExcelTemplateBO entity) -> entity.setTagNames(value));
        FIELD_SETTERS.put("年份标签", (String value, ExcelTemplateBO entity) -> entity.setYearTag(value));
        FIELD_SETTERS.put("演员", (String value, ExcelTemplateBO entity) -> entity.setActNames(value));
        FIELD_SETTERS.put("导演", (String value, ExcelTemplateBO entity) -> entity.setDirectorName(value));
    }

    public void processExcelData(String excelUrl, int batchSize) {
        try (InputStream inputStream = new URI(excelUrl).toURL().openStream()) {
            var excelListener = new CustomExcelListener(Math.max(MIN_BATCH_SIZE, Math.min(batchSize, MAX_BATCH_SIZE)));
            EasyExcel.read(inputStream, excelListener).sheet().doRead();
            excelListener.completionLatchAwait();
            log.info("Excel data processing completed successfully");
        } catch (Exception e) {
            log.error("Failed to process Excel data", e);
            throw new RuntimeException("Failed to process Excel data", e);
        }
    }

    private class CustomExcelListener extends AnalysisEventListener<Map<Integer, String>> {
        private final int batchSize;
        private final List<Map<Integer, String>> rawDataBuffer = new ArrayList<>();
        private Map<Integer, FieldSetter<ExcelTemplateBO>> columnMapping;
        private final CountDownLatch completionLatch = new CountDownLatch(1);
        private static final String END_MARKER = "__END__";
        private final ObjectMapper objectMapper = new ObjectMapper();

        public CustomExcelListener(int batchSize) {
            this.batchSize = batchSize;
            insertionExecutor.execute(() -> {
                while (!Thread.currentThread().isInterrupted()) {
                    List<String> datas = redisTemplate.opsForList().leftPop(IMPORT_EXCEL_QUEUE_KEY,
                            batchSize);
                    boolean isEnd = datas.isEmpty() || datas.getLast().equals(END_MARKER);
                    List<ArtWorkDO> batch = datas.stream().filter(data -> !data.equals(END_MARKER)).map(data -> {
                        try {
                            return objectMapper.readValue(data, ArtWorkDO.class);
                        } catch (JsonProcessingException e) {
                            redisTemplate.opsForList().rightPush(FAILED_DECODE_IMPORT_EXCEL_QUEUE_KEY, data);
                            log.error("Failed to deserialize JSON to ArtWorkDO", e);
                            return null;
                        }
                    }).filter(Objects::nonNull).toList();
                    if (!batch.isEmpty()) {
                        artworkMapper.insert(batch);
                    }
                    if (isEnd) {
                        completionLatch.countDown();
                        break;
                    }
                }
            });
        }

        public boolean completionLatchAwait() throws InterruptedException {
            return this.completionLatch.await(5, TimeUnit.MINUTES);
        }

        @Override
        public void invokeHeadMap(Map<Integer, String> headMap, AnalysisContext context) {
            columnMapping = new HashMap<>();
            headMap.forEach((index, header) -> {
                var fieldSetter = FIELD_SETTERS.get(header);
                if (fieldSetter == null) {
                    return;
                }
                columnMapping.put(index, fieldSetter);
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
                    var excelTemplateBO = new ExcelTemplateBO();
                    var artWorkDO = new ArtWorkDO();

                    data.forEach((index, value) -> {
                        FieldSetter<ExcelTemplateBO> fieldSetter = columnMapping.get(index);
                        if (fieldSetter == null || value == null || value.isEmpty()) {
                            return;
                        }
                        fieldSetter.setField(value, excelTemplateBO);
                    });
                    artWorkDO.setId(null);
                    artWorkDO.setTitle(excelTemplateBO.getTitle());
                    artWorkDO.setCover(excelTemplateBO.getCover());
                    artWorkDO.setLink(excelTemplateBO.getLink());
                    artWorkDO.setSubtitle(excelTemplateBO.getSubtitle());
                    artWorkDO.setSeason(excelTemplateBO.getSeasonValue());
                    artWorkDO.setEpisode(excelTemplateBO.getEpisodeValue());
                    artWorkDO.setCategoryId(excelTemplateBO.updateCategoryNameToCategoryId(artworkCategoryMapper));
                    artWorkDO.setTags(excelTemplateBO.updateTagNamesToTagIds(artworkTagMapper).toArray(new Integer[0]));
                    artWorkDO.setActs(excelTemplateBO.updateActNamesToActIds(peopleMapper, peopleTypeMapper)
                            .toArray(new Integer[0]));
                    artWorkDO.setYearTag(excelTemplateBO.getYear());

                    processedBatch.add(artWorkDO);
                }
                var artWorkRedisTemplate = new RedisTemplate<String, ArtWorkDO>()
                        .boundListOps(FAILED_ENCODE_IMPORT_EXCEL_QUEUE_KEY);
                redisTemplate.opsForList().rightPushAll(IMPORT_EXCEL_QUEUE_KEY,
                        processedBatch.stream().map(data -> {
                            try {
                                return objectMapper.writeValueAsString(data);
                            } catch (JsonProcessingException e) {
                                log.error("Failed to serialize batch to JSON", e);
                                artWorkRedisTemplate.rightPush(data);
                                return null;
                            }
                        }).filter(Objects::nonNull).toList());
            });
        }

        @Override
        public void doAfterAllAnalysed(AnalysisContext context) {
            if (!rawDataBuffer.isEmpty()) {
                processBatchInBackground(new ArrayList<>(rawDataBuffer));
                rawDataBuffer.clear();
            }
            try {
                processingExecutor.shutdown();
                if (!processingExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                    processingExecutor.shutdownNow();
                }
                redisTemplate.opsForList().rightPush(IMPORT_EXCEL_QUEUE_KEY, END_MARKER);
                insertionExecutor.shutdown();
                if (!insertionExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                    insertionExecutor.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.error("Interrupted while waiting for executor shutdown", e);
            }
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
            List<String> headers = new ArrayList<>(FIELD_SETTERS.keySet());
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
                    "喜剧,超能力,动画",
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
