package com.metawebthree.media.utils;

import java.util.ArrayList;
import java.util.List;

import org.commonmark.node.AbstractVisitor;
import org.commonmark.node.BlockQuote;
import org.commonmark.node.Code;
import org.commonmark.node.Emphasis;
import org.commonmark.node.FencedCodeBlock;
import org.commonmark.node.HardLineBreak;
import org.commonmark.node.Heading;
import org.commonmark.node.IndentedCodeBlock;
import org.commonmark.node.Link;
import org.commonmark.node.ListItem;
import org.commonmark.node.Node;
import org.commonmark.node.Paragraph;
import org.commonmark.node.SoftLineBreak;
import org.commonmark.node.StrongEmphasis;
import org.commonmark.node.Text;

import lombok.Data;

@Data
public class DefaultMarkdownVisitor extends AbstractVisitor {
    private List<MarkdownNode> nodes = new ArrayList<>();

    static public String getText(Node node) {
        return node instanceof Text ? ((Text) node).getLiteral()
                : node instanceof Link ? ((Link) node).getTitle() : "";
    }

    @Override
    public void visit(Text text) {
        nodes.add(new MarkdownNode("TEXT", text.getLiteral()));
        visitChildren(text);
    }

    @Override
    public void visit(Link link) {
        String url = link.getDestination();
        StringBuilder linkText = new StringBuilder();
        Node child = link.getFirstChild();
        while (child != null) {
            if (child instanceof Text) {
                linkText.append(((Text) child).getLiteral());
            }
            child = child.getNext();
        }

        nodes.add(new MarkdownNode("LINK", linkText.toString(), url));
        visitChildren(link);
    }

    @Override
    public void visit(Code code) {
        nodes.add(new MarkdownNode("INLINE_CODE", code.getLiteral()));
        visitChildren(code);
    }

    @Override
    public void visit(FencedCodeBlock codeBlock) {
        String language = codeBlock.getInfo();
        String codeContent = codeBlock.getLiteral();
        nodes.add(new MarkdownNode("CODE_BLOCK", codeContent, language, true));
        visitChildren(codeBlock);
    }

    @Override
    public void visit(IndentedCodeBlock codeBlock) {
        String codeContent = codeBlock.getLiteral();
        nodes.add(new MarkdownNode("CODE_BLOCK", codeContent, "", true));
        visitChildren(codeBlock);
    }

    @Override
    public void visit(Heading heading) {
        StringBuilder headingText = new StringBuilder();
        Node child = heading.getFirstChild();
        while (child != null) {
            if (child instanceof Text) {
                headingText.append(((Text) child).getLiteral());
            }
            child = child.getNext();
        }

        nodes.add(new MarkdownNode("HEADING_" + heading.getLevel(), headingText.toString()));
        visitChildren(heading);
    }

    @Override
    public void visit(Paragraph paragraph) {
        visitChildren(paragraph);
    }

    @Override
    public void visit(SoftLineBreak softLineBreak) {
        nodes.add(new MarkdownNode("LINE_BREAK", ""));
        visitChildren(softLineBreak);
    }

    @Override
    public void visit(HardLineBreak hardLineBreak) {
        nodes.add(new MarkdownNode("LINE_BREAK", ""));
        visitChildren(hardLineBreak);
    }

    @Override
    public void visit(Emphasis emphasis) {
        visitChildren(emphasis);
    }

    @Override
    public void visit(StrongEmphasis strongEmphasis) {
        visitChildren(strongEmphasis);
    }

    @Override
    public void visit(ListItem listItem) {
        StringBuilder itemText = new StringBuilder();
        Node child = listItem.getFirstChild();
        while (child != null) {
            if (child instanceof Text) {
                itemText.append(((Text) child).getLiteral());
            }
            child = child.getNext();
        }
        nodes.add(new MarkdownNode("LIST_ITEM", itemText.toString()));
        visitChildren(listItem);
    }

    @Override
    public void visit(BlockQuote blockQuote) {
        StringBuilder quoteText = new StringBuilder();
        Node child = blockQuote.getFirstChild();
        while (child != null) {
            if (child instanceof Text) {
                quoteText.append(((Text) child).getLiteral());
            }
            child = child.getNext();
        }
        nodes.add(new MarkdownNode("BLOCKQUOTE", quoteText.toString()));
        visitChildren(blockQuote);
    }
}
