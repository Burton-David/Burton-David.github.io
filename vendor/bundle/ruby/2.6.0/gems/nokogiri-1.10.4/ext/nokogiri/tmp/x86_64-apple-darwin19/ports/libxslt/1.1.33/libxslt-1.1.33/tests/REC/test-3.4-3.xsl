<xsl:stylesheet
        version="1.0"
        xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
>
    <xsl:strip-space elements="*"/>
    <xsl:template match="doc">
        <xsl:apply-templates select="*/*"/>
    </xsl:template>
</xsl:stylesheet>