package com.metawebthree.media;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.ibatis.cursor.Cursor;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.commonmark.node.Node;
import org.commonmark.parser.Parser;
import org.junit.Assert;
import org.junit.Test;
import org.junit.jupiter.api.Disabled;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.transaction.annotation.Transactional;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.toolkit.SqlHelper;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.media.DO.ArtWorkDO;
import com.metawebthree.media.DO.PeopleDO;
import com.metawebthree.media.DO.PeopleTypeDO;
import com.metawebthree.media.utils.DefaultMarkdownVisitor;

@RunWith(SpringRunner.class)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
@ComponentScan(basePackages = {
        "com.metawebthree.media",
        "com.metawebthree.common"
})
@EnableAutoConfiguration
public class MediaMapperTest {

    @Value("${test-md-file}")
    private String importMDFile;

    @Autowired
    private ArtWorkMapper artWorkMapper;

    @Autowired
    private PeopleMapper peopleMapper;

    @Autowired
    private PeopleTypeMapper peopleTypeMapper;

    @Test
    @Disabled("Fix data")
    public void fixArtWorkDirector() {
        String directorName = "文舟";
        String type = "director";

        PeopleTypeDO targetPeopleTypeDO = PeopleTypeDO.builder().type(type).build();
        PeopleTypeDO matchPeopleTypeDO = getOrCreatePeopleType(targetPeopleTypeDO);

        PeopleDO targetPeopleDO = PeopleDO.builder().name(directorName).types(new Short[] { matchPeopleTypeDO.getId() })
                .build();
        PeopleDO matchPeopleDO = getOrCreatePeople(targetPeopleDO);

        UpdateWrapper<ArtWorkDO> updateWrapper = new UpdateWrapper<ArtWorkDO>().eq("series", "骑士的沙丘").set("director",
                matchPeopleDO.getId());
        int result = artWorkMapper.update(updateWrapper);
        Assert.assertTrue(result > 0);
    }

    public PeopleTypeDO getOrCreatePeopleType(PeopleTypeDO targetPeopleTypeDO) {
        PeopleTypeDO matchPeopleTypeDO = peopleTypeMapper.selectOne(new MPJLambdaWrapper<PeopleTypeDO>()
                .select(PeopleTypeDO::getId).eq(PeopleTypeDO::getType, targetPeopleTypeDO.getType()));
        if (matchPeopleTypeDO.equals(null)) {
            int getTypeResult = peopleTypeMapper.insert(targetPeopleTypeDO);
            Assert.assertEquals(1, getTypeResult);
        }
        return matchPeopleTypeDO;
    }

    public PeopleDO getOrCreatePeople(PeopleDO targetPeopleDO) {
        MPJLambdaWrapper<PeopleDO> queryWrapper = new MPJLambdaWrapper<>();
        queryWrapper.select(PeopleDO::getId).eq(PeopleDO::getName, targetPeopleDO.getName());
        PeopleDO matchPeopleDO = peopleMapper.selectOne(queryWrapper);
        if (matchPeopleDO.equals(null)) {
            int result = peopleMapper.insert(matchPeopleDO);
            Assert.assertEquals(1, result);
        }
        return matchPeopleDO;
    }

    @Test
    @Disabled("Fix data")
    public void fixSubtitleUndefinedError() {
        String errorPart = "undefined";
        ArtWorkDO artWorkDO = ArtWorkDO.builder().subtitle(errorPart).build();
        UpdateWrapper<ArtWorkDO> updateWrapper = new UpdateWrapper<>();
        String fieldName = "subtitle";
        updateWrapper.likeRight(fieldName, artWorkDO.getSubtitle())
                .setSql(String.format("%s = REPLACE(%s, '%s', '')", fieldName, errorPart, fieldName));
        artWorkMapper.update(updateWrapper);
        ArtWorkDO errorColumn = artWorkMapper.selectOne(
                new MPJLambdaWrapper<ArtWorkDO>()
                        .select(ArtWorkDO::getId)
                        .like(ArtWorkDO::getSubtitle, "undefined"));
        Assert.assertNull(errorColumn);
    }

    @Test
    @Disabled("Fix data")
    public void fixArtworkSeriesError() throws IOException {
        Node document = analyzeFile();
        List<Integer> results = processArtworks(document);
        Assert.assertTrue(results.size() > 0);
        Assert.assertTrue(results.stream().allMatch(i -> i > 0));
    }

    public Node analyzeFile() throws IOException {
        Parser parser = Parser.builder().build();
        String fileContent = Files.readString(Path.of(importMDFile), java.nio.charset.StandardCharsets.UTF_8);
        Node document = parser.parse(fileContent);
        return document;
    }

    @Transactional
    public List<Integer> processArtworks(Node document) {
        List<Integer> results = new ArrayList<>();
        SqlSessionFactory factory = SqlHelper.sqlSessionFactory(ArtWorkDO.class);
        try (SqlSession session = factory.openSession()) {
            ArtWorkMapper mapper = session.getMapper(ArtWorkMapper.class);
            Cursor<ArtWorkDO> cursor = mapper.getCursor();
            List<ArtWorkDO> buffer = new ArrayList<>();
            int batchSize = 1000;
            Iterator<ArtWorkDO> iter = cursor.iterator();
            while (iter.hasNext()) {
                ArtWorkDO data = iter.next();
                data = processArtwork(data, document);
                buffer.add(data);
                if (buffer.size() >= batchSize) {
                    results.addAll(batchUpdate(buffer, mapper));
                    buffer.clear();
                }
            }
            if (!buffer.isEmpty()) {
                results.addAll(batchUpdate(buffer, mapper));
                buffer.clear();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return results;
    }

    private List<Integer> batchUpdate(List<ArtWorkDO> list, ArtWorkMapper artWorkMapper) {
        List<Integer> results = new ArrayList<>();
        for (ArtWorkDO u : list) {
            results.add(artWorkMapper.updateById(u));
        }
        return results;
    }

    private ArtWorkDO processArtwork(ArtWorkDO artWorkDO, Node document) {
        if (artWorkDO.getTitle() == null || artWorkDO.getTitle().isEmpty()) {
            ArtWorkDO pre = artWorkMapper.selectById(artWorkDO.getId() - 1);
            Node node = document.getNext();
            while (node != null && pre.getTitle() != null) {
                String text = DefaultMarkdownVisitor.getText(node);
                if (!text.contains(pre.getTitle().replaceFirst("1{1}$", ""))) {
                    continue;
                }
                Node cur = node.getNext();
                if (cur == null) {
                    continue;
                }
                String curText = DefaultMarkdownVisitor.getText(cur);
                artWorkDO.setTitle(curText);
            }
        }
        return artWorkDO;
    }
}
