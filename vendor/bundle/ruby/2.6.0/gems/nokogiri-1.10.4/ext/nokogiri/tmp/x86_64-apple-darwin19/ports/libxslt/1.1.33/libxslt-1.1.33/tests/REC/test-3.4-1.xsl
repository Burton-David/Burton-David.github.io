<xsl:stylesheet
        version="1.0"
        xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
        xmlns:foo1="urn:foo1"
>
    <xsl:strip-space elements="foo1:element1"/>
    <xsl:template match="doc">
        <xsl:apply-templates select="*/*"/>
    </xsl:template>
</xsl:stylesheet>
