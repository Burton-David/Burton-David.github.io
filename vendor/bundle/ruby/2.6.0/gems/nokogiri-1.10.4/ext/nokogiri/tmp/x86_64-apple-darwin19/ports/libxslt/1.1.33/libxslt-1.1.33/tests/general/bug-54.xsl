<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet
        version="1.0"
        xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
>
<xsl:output method="xml" version="1.0" encoding="UTF-8" indent="yes"/>

<xsl:template match="/all/*">
        <xsl:for-each select="namespace::*">
                <xsl:copy-of select="."/>
        </xsl:for-each>
</xsl:template>
<xsl:template match="/">
    <root>
    <xsl:apply-templates/>
    </root>
</xsl:template>

</xsl:stylesheet>
