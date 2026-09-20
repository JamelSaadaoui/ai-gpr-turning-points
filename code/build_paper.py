"""Build the self-contained article from the computed, frozen-sample results.

Run `python code/build_paper.py` from the package root after analysis/figures.
Equations and inline mathematics are editable native Office Math (OMML).
"""
from pathlib import Path
from datetime import datetime, timezone
import re
import math

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
OUT.mkdir(exist_ok=True)
FONT = "Palatino Linotype"
LABEL = {"gpr_ai":"Aggregate AI-GPR", "military_conflict":"Military conflict",
         "diplomatic_tension":"Diplomatic tension", "terrorism":"Terrorism",
         "civil_war":"Civil war", "nuclear_threat":"Nuclear threat", "coup":"Coup",
         "sanctions":"Sanctions", "other":"Other events"}
PRIMARY = ["military_conflict", "diplomatic_tension", "nuclear_threat"]
R = {p.stem: pd.read_csv(p) for p in (ROOT/"results").glob("*.csv")}


def row(name, **filters):
    d=R[name]
    for k,v in filters.items(): d=d[d[k].astype(str)==str(v)]
    if len(d)!=1: raise ValueError((name, filters, len(d)))
    return d.iloc[0]


def num(v,d=3):
    return f"{float(v):.{d}f}".replace("-", "−")


def pv(v):
    return "<0.001" if v<.001 else f"{v:.3f}"


def p_range(lo,hi):
    return "all <0.001" if hi<.001 else pv(lo)+" to "+pv(hi)


def pct(b): return 100*np.expm1(float(b)*np.log(1.1))


def mr(text, plain=False):
    r=OxmlElement("m:r")
    if plain:
        pr=OxmlElement("m:rPr"); st=OxmlElement("m:sty"); st.set(qn("m:val"),"p"); pr.append(st); r.append(pr)
    t=OxmlElement("m:t"); t.text=text; t.set(qn("xml:space"),"preserve"); r.append(t)
    return r


COMMANDS={"alpha":"α","beta":"β","gamma":"γ","delta":"δ","theta":"θ","pi":"π","chi":"χ",
          "rho":"ρ","lambda":"λ","Delta":"Δ","Gamma":"Γ","Sigma":"Σ","sum":"∑","in":"∈",
          "times":"×","cdot":"·","neq":"≠","geq":"≥","leq":"≤","approx":"≈",
          "varepsilon":"ε","epsilon":"ε","prime":"′","rightarrow":"→","infty":"∞",
          "quad":"  ","qquad":"    ","ldots":"…"}


def math_nodes(s):
    """Small deterministic parser for the notation used in this article."""
    i=0
    def group():
        nonlocal i
        if i<len(s) and s[i]=="{":
            i+=1; ans=parse("}"); i+=1; return ans
        return [atom()]
    def atom():
        nonlocal i
        if s[i]=="{":
            ns=group(); e=OxmlElement("m:box"); ee=OxmlElement("m:e"); [ee.append(n) for n in ns]; e.append(ee); return e
        if s[i]=="\\":
            i+=1; start=i
            while i<len(s) and s[i].isalpha(): i+=1
            c=s[start:i]
            if c=="widetilde":
                while i<len(s) and s[i]==" ": i+=1
                a=group(); acc=OxmlElement("m:acc"); pr=OxmlElement("m:accPr")
                ch=OxmlElement("m:chr"); ch.set(qn("m:val"),"̃"); pr.append(ch); acc.append(pr)
                e=OxmlElement("m:e"); [e.append(n) for n in a]; acc.append(e); return acc
            if c=="frac":
                a=group(); b=group(); f=OxmlElement("m:f")
                for tag,ls in [("m:num",a),("m:den",b)]:
                    e=OxmlElement(tag); [e.append(n) for n in ls]; f.append(e)
                return f
            if c in ("log","ln","exp","Cov","E","Var","rank","max","min"):
                return mr(c,True)
            if not c:
                c=s[i]; i+=1; return mr(c)
            if c not in COMMANDS: raise ValueError("Unsupported math command: "+c)
            return mr(COMMANDS[c])
        c=s[i]; i+=1; return mr(c)
    def parse(stop=None):
        nonlocal i
        out=[]
        while i<len(s) and s[i]!=stop:
            base=atom(); scripts={}
            while i<len(s) and s[i] in "_^":
                kind=s[i]; i+=1; scripts[kind]=group()
            if scripts:
                tag="m:sSubSup" if len(scripts)==2 else ("m:sSub" if "_" in scripts else "m:sSup")
                node=OxmlElement(tag); e=OxmlElement("m:e"); e.append(base); node.append(e)
                for kind,tag2 in [("_","m:sub"),("^","m:sup")]:
                    if kind in scripts:
                        e=OxmlElement(tag2); [e.append(n) for n in scripts[kind]]; node.append(e)
                base=node
            out.append(base)
        return out
    return parse()


def append_math(p,s,display=False):
    m=OxmlElement("m:oMath"); [m.append(n) for n in math_nodes(s)]
    if display:
        wrapper=OxmlElement("m:oMathPara"); wrapper.append(m); p._p.append(wrapper)
    else: p._p.append(m)


def rich(p,s):
    s=s.replace("p=<", "p<")
    for i,part in enumerate(s.split("$")):
        if i%2: append_math(p,part)
        else: p.add_run(part)
    return p


doc=Document(); sec=doc.sections[0]
sec.page_width=Inches(8.5); sec.page_height=Inches(11)
sec.top_margin=Inches(.8); sec.bottom_margin=Inches(.8)
sec.left_margin=Inches(.87); sec.right_margin=Inches(.87)
for st in doc.styles:
    if hasattr(st,"font"):
        st.font.name=FONT; st.font.color.rgb=RGBColor(0,0,0)
        st._element.get_or_add_rPr().rFonts.set(qn("w:ascii"),FONT)
        st._element.rPr.rFonts.set(qn("w:hAnsi"),FONT)
        for attr in list(st._element.rPr.rFonts.attrib):
            if "theme" in attr.lower(): del st._element.rPr.rFonts.attrib[attr]
        for key in ("ascii", "hAnsi", "cs", "eastAsia"):
            st._element.rPr.rFonts.set(qn("w:"+key),FONT)
doc.styles["Normal"].font.size=Pt(11)
doc.styles["Normal"].paragraph_format.line_spacing=1.12
doc.styles["Normal"].paragraph_format.space_after=Pt(6)
for name,size in [("Title",19),("Heading 1",14),("Heading 2",12),("Caption",10)]:
    doc.styles[name].font.size=Pt(size)
    doc.styles[name].font.bold=name!="Caption"
    doc.styles[name].font.italic=False
doc.core_properties.author="Jamel Saadaoui"
doc.core_properties.last_modified_by="Jamel Saadaoui"
doc.core_properties.title="Not All Geopolitical Turning Points Are Alike: Oil Price Effects and Implications for Energy Security Policy"
doc.core_properties.subject="Event-specific geopolitical risk and lag-augmented instrumental-variable local projections"
doc.core_properties.comments=""
doc.core_properties.created=datetime.now(timezone.utc)
doc.core_properties.modified=datetime.now(timezone.utc)
footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
fld=OxmlElement("w:fldSimple"); fld.set(qn("w:instr"),"PAGE"); footer._p.append(fld)


def para(s,style=None):
    p=doc.add_paragraph(style=style); rich(p,s)
    p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def heading(s,level=1): return doc.add_heading(s,level)


def equation(s,n):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(6); p.paragraph_format.space_after=Pt(9)
    append_math(p,s,True); p.add_run(f"   ({n})")
    return p


def note(s):
    p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(10)
    p.paragraph_format.keep_together=True
    r=p.add_run("Notes: "); r.bold=True
    rich(p,s)
    for rr in p.runs: rr.font.size=Pt(9)
    return p


def caption(s,keep=True):
    p=doc.add_paragraph(style="Caption"); rich(p,s)
    p.paragraph_format.keep_with_next=keep; p.paragraph_format.space_before=Pt(9)
    p.paragraph_format.space_after=Pt(4)
    return p


def table(title,headers,rows,widths,notes,font=9.3):
    caption(title)
    t=doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    widths=[w*6.70/sum(widths) for w in widths]
    for col,w in zip(t.columns,widths): col.width=Inches(w)
    pr=t._tbl.tblPr; borders=OxmlElement("w:tblBorders")
    for edge in ["top","bottom","left","right","insideH","insideV"]:
        e=OxmlElement("w:"+edge); e.set(qn("w:val"),"single" if edge in ("top","bottom") else "nil")
        e.set(qn("w:sz"),"8"); e.set(qn("w:color"),"000000"); borders.append(e)
    pr.append(borders)
    for ri,values in enumerate([headers]+rows):
        cells=t.rows[0].cells if ri==0 else t.add_row().cells
        for ci,value in enumerate(values):
            c=cells[ci]; c.width=Inches(widths[ci]); c.vertical_alignment=WD_ALIGN_VERTICAL.CENTER
            p=c.paragraphs[0]; p.paragraph_format.space_after=Pt(3); p.paragraph_format.space_before=Pt(3)
            p.paragraph_format.line_spacing=1
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
            rich(p,str(value))
            for r in p.runs: r.font.size=Pt(font); r.bold=ri==0
            cp=c._tc.get_or_add_tcPr(); mar=OxmlElement("w:tcMar")
            for side in ("top","bottom","start","end"):
                e=OxmlElement("w:"+side); e.set(qn("w:w"),"55"); e.set(qn("w:type"),"dxa"); mar.append(e)
            cp.append(mar)
            if ri==0:
                cb=OxmlElement("w:tcBorders"); e=OxmlElement("w:bottom"); e.set(qn("w:val"),"single"); e.set(qn("w:sz"),"5"); cb.append(e); cp.append(cb)
        rp=t.rows[ri]._tr.get_or_add_trPr(); rp.append(OxmlElement("w:cantSplit"))
        if ri==0: rp.append(OxmlElement("w:tblHeader"))
    rows_to_keep = t.rows if title.startswith(("Table B1", "Table 5")) else [t.rows[-1]]
    for kept_row in rows_to_keep:
        for c in kept_row.cells:
            for p in c.paragraphs: p.paragraph_format.keep_with_next=True
    note(notes)
    return t


def figure(filename,title,notes,width=6.5,newpage=False):
    if newpage: doc.add_page_break()
    path=ROOT/"figures"/(filename+".png")
    if not path.exists(): raise FileNotFoundError(path)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next=True
    p.add_run().add_picture(str(path),width=Inches(width))
    caption(title); note(notes)


def beta(out,event,h,model="joint"):
    return row("baseline_iv",model=model,outcome=out,event=event,horizon=h)


heading_text="Not All Geopolitical Turning Points Are Alike: Oil Price Effects and Implications for Energy Security Policy"
doc.add_paragraph(heading_text,"Title")
p=doc.add_paragraph("William Ginn and Jamel Saadaoui")
p.paragraph_format.space_after=Pt(4)
p=doc.add_paragraph("William Ginn: Labcorp, Artificial Intelligence, USA; Coburg University of Applied Sciences, Germany")
p.paragraph_format.space_after=Pt(2)
p=doc.add_paragraph("Jamel Saadaoui: Université Paris 8 Vincennes–Saint-Denis, LED, France; Australian National University, Centre for Applied Macroeconomic Analysis, Australia")
p.paragraph_format.space_after=Pt(12)
heading("Abstract",2)
dip=beta("lbrent","diplomatic_tension",24); nuc=beta("lbrent","nuclear_threat",24)
para(f"Energy-security decisions often treat geopolitical risk as a common source of upward oil-price pressure. This paper tests that premise using monthly event-specific AI-GPR indices, real Brent and WTI prices, and oil-market fundamentals. The established turning-point identification strategy instruments each geopolitical index with its raw second difference, emphasizing abrupt changes in trajectory. Lag-augmented instrumental-variable local projections reveal different medium-run responses across event types. In the joint Brent model, a 10% increase in one plus the relevant index is associated with an estimated {abs(pct(dip.beta)):.2f}% price decline after 24 months for diplomatic tension and a {pct(nuc.beta):.2f}% increase for nuclear threat, conditional on the identifying assumptions. Equality of the principal category responses is rejected at medium horizons, and Sanderson–Windmeijer statistics show that every principal conditional first stage is exceptionally strong. Lead-curvature placebos do not reject for the principal categories, event-window exclusions preserve the main patterns, and the positive nuclear response remains stable in samples ending before 2012 and before 2019. The policy implication is diagnostic rather than mechanical: agencies should combine event-specific geopolitical monitoring with physical-market indicators before interpreting an escalation as an oil-supply emergency.")
para("Keywords: geopolitical risk; energy security; oil prices; strategic reserves; instrumental variables; local projections.")
para("JEL classification: C26; C32; F51; Q41.")

heading("1 Introduction")
para("Energy authorities must often decide whether a geopolitical escalation signals an imminent supply disruption, a deterioration in prospective energy demand, or primarily a change in risk perceptions. The distinction matters for emergency-stock readiness, market communication, and the scenarios used to assess inflation and energy affordability. A response appropriate to a threatened physical shortage may be unnecessary, or even counterproductive, when the dominant signal is weaker expected global activity.")
para("An escalation of geopolitical risk can raise oil prices through fears of supply disruption, yet lower them through weaker expectations of global activity. The balance depends on the event. A nuclear confrontation, a diplomatic dispute, and an armed conflict do not convey the same information about production capacity, trade, or future demand. Aggregating these developments into one risk index can therefore conceal economically different responses, including effects with opposite signs, and provide an ambiguous signal for energy policy.")
para("This paper examines that heterogeneity using monthly event-specific indices from the AI-based geopolitical risk database of Iacoviello and Tong (2026). The analysis covers the aggregate index and eight categories, with military conflict, diplomatic tension, and nuclear threat forming the principal comparison. Real oil-price levels are estimated from impact to four years, separately for Brent and WTI. The joint specification includes the three principal categories simultaneously, allowing the response to one category to condition on variation in the others.")
para("The turning-point strategy developed in Saadaoui (2026, Journal of Comparative Economics) distinguishes gradual geopolitical evolution from abrupt changes in trajectory. Rather than interpreting every movement in the level of geopolitical risk as exogenous, the instrument uses the second difference of the underlying index. This transformation removes an affine trajectory and gives prominence to abrupt accelerations or decelerations. The methodological contribution of the present paper is therefore not to introduce a new estimator or instrument design. It applies an established identification strategy to a new global, event-specific measurement framework and to the energy market. Economically, a sudden strategic escalation or an unexpected cessation of escalation is more plausibly driven by geopolitical decisions than by routine monthly oil-market news, conditional on recent oil prices, global activity, production growth, and geopolitical dynamics.")
para("The exclusion argument concerns the economic state represented by the index, not a causal influence of a numerical transformation in isolation. The selected turning-point variation operates through a change in geopolitical conditions and their consequences for economic expectations and decisions. When that change alters expected oil demand, precautionary inventory demand, or compensation for geopolitical risk, these are transmission mechanisms of the geopolitical state. They are not, merely by being economic mechanisms, separate violations of exclusion. For the event-specific application, the maintained restriction requires the corresponding risk measure to capture the market-relevant geopolitical variation sufficiently well for no independent component of the instrument to remain in the oil-price disturbance.")
para("The empirical design distinguishes that justification from the questions answered by diagnostics. The first stage establishes whether the instrument contains usable conditional variation. Lead-curvature regressions examine whether future turning-point measures are related to oil prices before their dated month. Event exclusions show whether a small set of episodes dominates the estimates. Lag-augmented inference and sample splits quantify further limits. None of these exercises is a substitute for specifying the economic source of variation and the channel through which it is allowed to affect prices.")
para(f"The principal full-sample finding is a medium-run divergence. In the joint Brent model, the 24-month coefficient on diplomatic tension is {num(dip.beta)}, while that on nuclear threat is {num(nuc.beta)}. The military-conflict response is positive at impact but small and imprecise at 24 months. Tests of equality establish that the medium-run differences are not based solely on visually comparing separate confidence intervals. The descriptive subsample evidence is less uniform: the negative diplomatic response persists in sign, whereas the nuclear response changes substantially. Accordingly, the paper does not interpret every category or period as supporting one invariant geopolitical multiplier.")
para("The contribution to energy policy is to replace an undifferentiated geopolitical alarm signal with an event-specific diagnostic grounded in a transparent econometric design. The estimates do not identify the effect of releasing strategic stocks, imposing sanctions, or changing monetary policy. They indicate when the geopolitical composition of news is consistent with different oil-price risks. Event-specific indices can therefore guide scenario design and the intensity of monitoring, while physical production, trade, shipping, inventory, and futures-market evidence determine whether intervention is warranted.")

heading("2 Related literature")
heading("2.1 Measuring geopolitical conditions and uncertainty",2)
para("Geopolitical risk is a multidimensional economic object. Threats of conflict, realized violence, sanctions, and diplomatic confrontation can convey different information even when they occur during the same episode. Caldara and Iacoviello (2022) make this distinction operational with a news-based index and separate measures of geopolitical threats and acts. Their contribution provides a systematic alternative to selecting a few celebrated crises. Iacoviello and Tong (2026) extend this measurement agenda through AI-assisted classification and scoring. Event-specific indices permit comparisons that an aggregate index cannot reveal, but remain measures of reported geopolitical conditions rather than externally identified structural innovations.")
para("The broader uncertainty literature explains why these distinctions matter. Bloom (2009) studies how uncertainty can lead firms to postpone investment and employment decisions. Baker, Bloom, and Davis (2016) construct newspaper-based measures of economic policy uncertainty. Geopolitical risk and policy uncertainty can move together, yet they need not represent the same disturbance. A military threat may alter beliefs about a rare disruption, while a diplomatic disagreement may change the expected persistence of commercial restrictions. This paper uses the event classification to examine heterogeneous price responses rather than interpreting all uncertainty as a single shock.")
para("Trade-policy research gives economic content to the diplomatic channel. Handley and Limão (2017) show how uncertainty about trade policy affects entry, investment, trade, and welfare. Caldara et al. (2020) document adverse investment and activity responses to increases in trade-policy uncertainty. These mechanisms can reduce prospective energy demand when diplomatic relations deteriorate. Their relevance does not imply that the present regressions estimate a trade-policy shock or identify a trade mediator separately. They explain why a geopolitical deterioration can lower the oil price even in the absence of an immediate increase in oil supply.")
para("Bilateral relationship measures provide complementary information about the political state. Mignon and Saadaoui (2024) examine political tensions, geopolitical risks, and oil prices, while Mignon and Saadaoui (2025) study distributional asymmetries using quantile methods. Saadaoui (2026) develops and implements turning-point identification for bilateral political relations. The present application transfers that published strategy to global event-specific risk measures and the oil market. Its contribution is neither a new raw news index nor a new identification method. It establishes whether oil-price responses differ across geopolitical dimensions when abrupt trajectory changes supply the identifying variation.")
heading("2.2 Oil market mechanisms and geopolitical transmission",2)
para("The oil literature cautions against assigning a unique structural interpretation to a price movement. Kilian (2009) distinguishes disturbances to oil supply, global aggregate demand, and oil-specific demand, showing that their economic implications differ. Baumeister and Kilian (2016) place major historical fluctuations in this framework and explain why oil-price surprises depend on the information available to different market participants. The implication for geopolitical analysis is direct: an event label does not determine the sign of the equilibrium price response. The balance between expected demand, available supply, and precautionary behavior matters.")
para("Expectations can move spot prices before physical quantities adjust. Alquist and Kilian (2010) analyze the information contained in oil futures and the role of uncertainty about future scarcity. Kilian and Murphy (2014) incorporate inventory information to distinguish speculative demand from other oil-market forces. A nuclear threat can therefore affect oil prices through precautionary inventory demand or the valuation of disruption risk even when current production remains unchanged. Conversely, weaker expected activity after a diplomatic rupture can reduce oil demand. These are plausible mechanisms for the signs observed below; inventories, futures premia, and expectations are not separately estimated in this paper.")
para("Structural identification in the oil market depends on substantive restrictions. Caldara, Cavallo, and Iacoviello (2019) show how the choice of supply and demand elasticities affects the identification of oil shocks. Baumeister and Hamilton (2019) develop an approach that makes incomplete identifying information explicit. Their contributions motivate including world activity and oil-production dynamics, while avoiding the claim that those controls alone recover structural supply and demand shocks. Adding contemporaneous production growth is reported only as a change in the conditioning set, since production may itself be part of geopolitical transmission.")
para("Recent work by Verduzco-Bustos and Zanetti (2026) studies geopolitical oil-price shocks directly. The question here is complementary: whether different geopolitical states produce distinguishable response profiles under a common dynamic specification. The joint model conditions military conflict, diplomatic tension, and nuclear threat on one another. Its coefficients differ in interpretation from the separate-category estimates, which also reflect co-movement with omitted geopolitical dimensions. Comparing the two therefore informs interpretation, rather than supplying interchangeable estimates of an identical intervention.")
heading("2.3 Energy security policy and information",2)
para("Energy security is not a single observable condition. Winzer (2012) organizes it around the risks that interrupt energy continuity, while Cherp and Jewell (2014) emphasize the vulnerability of vital energy systems. These perspectives imply that policymakers require information about the source, probability, exposure, and likely duration of a disturbance, not merely a high aggregate risk reading. For oil-importing economies, the same geopolitical headline can imply different combinations of availability risk, price risk, and macroeconomic demand risk.")
para("This distinction is central to the use of emergency oil stocks and related contingency instruments. A news-based indicator cannot establish that a physical shortage exists, and the present estimates do not evaluate a reserve release. Event-specific geopolitical measures can nevertheless improve the diagnostic stage that precedes a policy decision. A category associated with precautionary or disruption-related price pressure can justify intensified surveillance and scenario preparation; a category associated with weaker demand expectations calls for a different assessment. In both cases, geopolitical signals should be corroborated with production, export, shipping, inventory, and forward-market information.")
para("The paper therefore contributes to energy-security policy through classification and timing. It estimates whether distinct geopolitical trajectories carry different oil-price implications and how those implications evolve over the policy horizon. The objective is not to derive a mechanical intervention threshold. It is to reduce the risk that decision-makers map every geopolitical escalation into the same presumed oil-supply shock.")
heading("2.4 Dynamic estimation and inference",2)
para("Jordà (2005) proposes local projections as direct horizon-specific regressions. Jordà and Taylor (2025) survey their use for dynamic causal analysis, including identification, inference, and practical specification choices. Plagborg-Møller and Wolf (2021) establish the population connection between local projections and vector autoregressions. Flexibility in the response profile should not be confused with an absence of identifying assumptions: the treatment, instrument, information set, and interpretation of a unit intervention still need to be specified. Stock and Watson (2018) and Nakamura and Steinsson (2018) provide the relevant framework for instrument-based identification of macroeconomic effects.")
para("Finite-sample precision and specification stability remain relevant. Barnichon and Brownlees (2019) propose smooth local projections to improve precision by imposing structure across horizons. Li, Plagborg-Møller, and Wolf (2024) compare the bias and variance of alternative estimators across many data-generating processes. Inoue, Rossi, and Wang (2024) develop local projections for unstable environments. The present estimates are unsmoothed and constant within each estimation sample. Their wide bands at some horizons and the descriptive subsample differences are reported directly; the sample splits do not constitute estimation of a time-varying parameter model.")
para("The baseline inference follows the lag-augmentation approach rather than appending a Newey–West correction to the augmented regression. Montiel Olea and Plagborg-Møller (2021) show that, under their conditions, lag-augmented local projections admit heteroskedasticity-robust inference despite serial correlation in the projection residuals. The relevant object is the regression score. White (1980) provides the heteroskedasticity-consistent sandwich framework, while MacKinnon and White (1985) study finite-sample leverage corrections. The baseline uses HC3 with an explicit IV implementation. Conventional two-price-lag projections with Newey and West (1987) covariance estimates are estimated separately as a comparison, following the distinction emphasized by Jordà and Taylor (2025).")
para("Instrument strength and instrument validity answer different questions. Staiger and Stock (1997) show why conventional IV inference can be unreliable when relevance is weak, while Sanderson and Windmeijer (2016) study conditional first stages in models with multiple endogenous regressors. These results motivate reporting both ordinary first-stage relevance statistics and formal conditional-strength diagnostics. In the present application, the resulting statistics are far above conventional weak-instrument ranges; the substantive identification discussion therefore concerns exclusion and economic interpretation rather than weak relevance.")
para("Two further distinctions govern interpretation. Angrist and Imbens (1995) explain how IV can identify instrument-dependent average causal responses with variable treatment intensity under additional assumptions. With continuous news indices, the coefficients here are described as local to the instrument-selected variation, without asserting a binary complier population or universally positive weights. Montiel Olea and Plagborg-Møller (2019) distinguish confidence statements about an entire response path from pointwise intervals. The paper consequently reports formal category-equality tests, selected-path tests, and approximate simultaneous bands alongside the main HC3 curves.")
para("The contribution brings these strands together in one event-specific oil-price analysis. The curvature filter selects changes in geopolitical trajectories; the economic exclusion argument describes how that variation reaches oil prices; and the separate and joint projections quantify the resulting dynamic associations under the maintained restrictions. This design makes the geopolitical composition of the response explicit. It does not replace structural oil-market models, nor claim that every geopolitical category has equal instrument strength, identical timing properties, or stable effects across historical periods.")

heading("3 Data and geopolitical turning points")
heading("3.1 Measurement and sample",2)
para("The monthly calendar runs from January 1990 to August 2026. AI-GPR supplies an aggregate risk index and eight event types: military conflict, diplomatic tension, terrorism, civil war, nuclear threat, coup or regime change, sanctions, and other events. These variables measure score-weighted news coverage, not counts of independent geopolitical incidents. The category series share the source's normalization; they are not each rescaled to have mean 100. The source vintage used here contains observations through August 2026. Appendix A records sources, transformations, and endpoint differences.")
para("Brent and WTI are monthly spot-price measures deflated by the U.S. consumer price index and expressed in natural logarithms. Global activity is measured by the Baumeister–Hamilton world industrial production series. The production control is the first difference of log global oil production. Because the BLS did not report CUSR0000SA0 for October 2025, its level is log-linearly interpolated between September and November 2025; nominal Brent and WTI remain observed EIA values. No other observation is filled. World industrial production ends in June 2026 and oil production in February 2026. These endpoints differ from the last available GPR observation and determine usable regression origins through the specified lags.")
para("Every lag and lead is constructed on the complete monthly calendar before missing observations are removed. Each horizon then uses only the variables entering that equation. Consequently, the impact regression has 428 complete origins, ending in March 2026; the 24-month regression has 412 origins and the 48-month regression 388. The sample at a positive horizon does not additionally require the contemporaneous oil price unless it appears in the specified regressors. This convention avoids dropping observations for a variable the equation does not use. Covariance calculations preserve the original calendar distances across missing months.")
para("Figure 1 displays all observed level variables and the transformed geopolitical indices. Table 1 summarizes both the estimation variables and the raw second differences. The logged production level is shown to document the source series; the regressions use its monthly growth rate. Comparisons of variability across event categories should respect their common source units and differing means. A unit increase in a rare-event index is not automatically economically equivalent to a unit increase in the aggregate index.")
rows=[]
for _,r in R["summary_statistics"].iterrows():
    label=r.label.replace("Log(1 + ","ln(1 + ").replace("Second difference:","Δ²")
    rows.append([label,str(int(r.n)),num(r["mean"]),num(r.sd),num(r["min"]),num(r["max"])])
table("Table 1  Summary statistics",["Variable","N","Mean","SD","Minimum","Maximum"],rows,[2.75,.45,.8,.8,.95,.95],"Statistics use available observations from January 1990 to August 2026 before regression-specific lag and lead restrictions. Prices, industrial production, and production levels are natural logarithms. Geopolitical regressors are $x_{j,t}=\\ln(1+G_{j,t})$; instruments are raw-index second differences $Z_{j,t}=\\Delta^2G_{j,t}$. Oil-production growth is $\\Delta\\ln Q_t$. Missing observations are retained as missing.",font=8.8)
figure("fig01_all_variables","Fig. 1.  Variables entering the analysis","Panels (a)–(e) show log real WTI, log real Brent, log world industrial production, log global oil production, and oil-production growth. Panels (f)–(n) show the natural logarithm of one plus aggregate AI-GPR, military conflict, diplomatic tension, terrorism, civil war, nuclear threat, coup, sanctions, and other events, respectively. The production level is displayed for reference; its first difference enters the local projections. Monthly observations extend to August 2026 where available. The unavailable October 2025 CPI deflator is log-linearly interpolated, producing complete real Brent and WTI histories; no nominal oil-price observation is filled.",width=6.35)
heading("3.2 The turning point filter",2)
para("Let $G_{j,t}$ denote the raw geopolitical index for category $j$ in month $t$. Define the endogenous regressor and instrument separately:")
equation(r"x_{j,t}=\ln(1+G_{j,t}),\qquad Z_{j,t}=\Delta^2G_{j,t}=G_{j,t}−2G_{j,t−1}+G_{j,t−2}",1)
para("A positive $Z_{j,t}$ means that the index's monthly change has increased relative to the previous month. It can represent an intensification of rising risk or a deceleration of falling risk. It need not imply that the level increased. A negative value has the corresponding interpretation. This distinction is important when associating an instrument spike with an event: the month after a large increase can generate a large negative second difference as escalation subsides, even while geopolitical risk remains elevated.")
para("An affine component of the raw index has zero second difference. More generally, the transformation attenuates smooth evolution and places relatively greater weight on high-frequency changes in trajectory. The economic purpose is to emphasize event-driven reorientations that are less plausibly induced by routine monthly macroeconomic feedback. It is not a threshold rule that retains only a handpicked list of shocks. All complete-case months contribute to estimation, with their contribution determined by the instrument after conditioning on the included controls. Figure 2 makes that distinction visible.")
para("The first-stage relationship is assessed empirically. The endogenous variable is logarithmic, while the instrument is the second difference of the raw index. Conditional relevance depends on their covariance after removing the controls, and may differ across categories and horizon-specific samples. It is therefore neither assumed to be strong for every category nor inferred from an identity between the two series. A separate diagnostic using the second difference of the logged regressor is described below; with its own two lags included, that alternative is algebraically equivalent to ordinary least squares and is not presented as an independent IV validation.")
figure("fig02_turning_points","Fig. 2.  Raw index geopolitical turning points","Panels show $Z_{j,t}=G_{j,t}−2G_{j,t−1}+G_{j,t−2}$ for aggregate AI-GPR and the eight event types, in the same order as the geopolitical panels of Fig. 1. Units are source-index points. Positive and negative values measure accelerations and decelerations, not necessarily increases and decreases in the index level. No cutoff is used to select observations.")

heading("4 Identification and econometric specification")
heading("4.1 Economic interpretation of the identifying variation",2)
para("The strategy identifies variation associated with changes in the geopolitical trajectory, conditional on the recent macroeconomic and geopolitical history. Its economic defense has two parts. First, routine monthly oil-price or activity innovations are less likely to generate abrupt shifts in strategic behavior than to co-move with gradual changes in geopolitical conditions. Second, the market-relevant geopolitical state carries the information that changes expectations and economic decisions. An escalation can alter expected activity, anticipated restrictions, precautionary demand, and risk compensation. These consequences are allowed by the model because they transmit the change in geopolitical conditions into prices.")
para("This interpretation is particularly relevant to a trade-war announcement or a sudden diplomatic rupture. The event changes the expected trajectory of the relationship and therefore the economic outlook associated with that relationship. The identifying claim is that the selected turning-point variation reaches oil prices through that change, including its consequences for expected demand and risk. The presence of an expectations channel is not evidence of an additional independent channel. What the exclusion restriction rules out is a component of the instrument that moves oil prices while bypassing the geopolitical state represented by the treatment.")
para("In the present global-risk application, let the event index proxy the market-relevant geopolitical state of its category. Conditional on the controls, its raw curvature must be orthogonal to oil-price innovations unrelated to the instrumented change in that state. This requirement is economically more demanding for some categories than others. Military events may simultaneously affect production facilities; news coverage may react to economic developments; several types of risk may move together. The joint model addresses observed co-movement among its three included states. It does not make every remaining geopolitical or measurement component disappear. The interpretation therefore remains category-specific, with the economic sufficiency of the measured state an explicit maintained assumption.")
para("The argument is about the source and transmission of variation, rather than a requirement that financial markets wait one full month before reacting. Markets may respond within the announcement month through the information conveyed about geopolitical conditions. Monthly observations cannot establish a finer within-month ordering. The timing diagnostics reported below investigate pre-event associations at the monthly frequency, while the exclusion restriction governs whether the instrument's contemporaneous and subsequent effects operate through the instrumented state. Filtering and timing evidence support this interpretation in different ways; neither is equated with a mathematical proof of the restriction.")

heading("4.2 Lag augmented local projections",2)
para("Let $p_t$ denote the log real oil price, $w_t$ log world industrial production, and $q_t=\\Delta\\ln Q_t$ oil-production growth. For a single category, the outcome equation at horizon $h$ is")
equation(r"p_{t+h}=\alpha_h+\beta_{j,h}x_{j,t}+\gamma_h^{\prime}C_{j,t}+u_{t+h},\quad h=0,\ldots,48",2)
para("The control vector is written explicitly to make the lag augmentation and information set unambiguous:")
equation(r"C_{j,t}=(w_{t−1},w_{t−2},q_{t−1},q_{t−2},p_{t−1},p_{t−2},p_{t−3},x_{j,t−1},x_{j,t−2})^{\prime}",3)
para("The baseline therefore contains two lags of activity, production growth, and the geopolitical regressor, plus three oil-price lags. The third price lag is the augmentation. There is no contemporaneous production control in the baseline. The dependent variable is a future log-price level, not a cumulative sum of price changes. Holding the included lagged information fixed, $\\beta_{j,h}$ describes the response to the current instrument-induced movement in the geopolitical regressor and the subsequent evolution associated with that movement.")
para("The first stage is re-estimated on the same complete observations as each horizon's outcome regression:")
equation(r"x_{j,t}=a_{j,h}+\pi_{j,h}Z_{j,t}+\lambda_{j,h}^{\prime}C_{j,t}+v_{j,t}",4)
para("A tilde denotes the residual from projection on the constant and controls. The scalar IV coefficient and identifying moment are")
equation(r"\beta_{j,h}^{IV}=\frac{E[\widetilde Z_{j,t}\widetilde p_{t+h}]}{E[\widetilde Z_{j,t}\widetilde x_{j,t}]},\qquad E[\widetilde Z_{j,t}u_{t+h}]=0",5)
para("The denominator must be nonzero. The numerator incorporates all economic responses allowed to follow from the instrumented geopolitical variation. If effects are heterogeneous, the coefficient is local to the variation weighted by the instrument and included controls. It need not equal the average effect of every geopolitical event. This is an instrument-specific local causal interpretation; the binary-treatment complier definition of LATE does not apply automatically to a continuous news index without additional assumptions about heterogeneity and the first stage.")
para("For the joint model, $X_t$ stacks military conflict, diplomatic tension, and nuclear threat, each transformed by $\\ln(1+G)$. The three corresponding raw second differences form $Z_t$. The control vector $C_t$ contains two lags of every included geopolitical regressor and the same oil-fundamental and price lags:")
equation(r"p_{t+h}=\alpha_h+B_h^{\prime}X_t+\Gamma_h^{\prime}C_t+u_{t+h},\qquad E[\widetilde Z_tu_{t+h}]=0",6)
para("All three current category variables are treated as endogenous and instrumented jointly. Identification requires the residualized instrument–regressor covariance matrix to have full rank. Both the scalar and joint models are exactly identified, so an overidentification test is unavailable. Single-category estimates condition on their own history but do not hold other current categories fixed; joint estimates do. They therefore answer related but distinct questions.")
para("To compare magnitudes transparently, the figures report responses to a 10% increase in one plus the source index. This corresponds to an increase of $\\ln(1.10)$ in the endogenous regressor. Figure units are log points multiplied by 100. Exact percentage changes in the geometric price response are reported in the main results table:")
equation(r"r_{j,h}=100\beta_{j,h}\ln(1.10),\qquad e_{j,h}=100[\exp\{\beta_{j,h}\ln(1.10)\}−1]",7)
para("This normalization is not a one-standard-deviation shock and does not imply equal physical severity across categories. When an index is near zero, a 10% increase in one plus that index differs materially from a 10% increase in the index itself. The distinction is retained in every figure note and in the interpretation of coefficient-equality tests.")

heading("4.3 Anticipation and statistical inference",2)
para("The anticipation exercise fixes the turning-point regressor at $t+2$ and examines oil prices at $t$ and $t+1$. For each category, and for the joint set, the reduced-form placebo equation is")
equation(r"p_{t+h}=a_h+\theta_h^{\prime}Z_{t+2}+d_h^{\prime}C_t+e_{t+h},\qquad h\in\{0,1\}",8)
para("The future regressor remains $Z_{t+2}$ in both equations. The controls are those predetermined at origin $t$; they are not shifted forward. The individual null is $\\theta_h=0$, and the two-horizon category null sets both coefficients to zero. In the joint model, an additional six-restriction test sets all three category coefficients at both horizons to zero. Outcomes at $t+2$ and later are not called anticipation tests because they no longer precede the dated future turning point.")
para("These are lead-curvature placebos, not event-study leads of a separately observed structural shock. In particular, $Z_{j,t+2}=G_{j,t+2}−2G_{j,t+1}+G_{j,t}$ includes current and next-month index levels. Their association with prices can generate a nonzero placebo coefficient even without advance knowledge of the later geopolitical innovation. Conversely, a failure to reject may reflect limited power. The tests are useful timing diagnostics for this exact transformation; the paper does not interpret them as proving that every relevant announcement was unanticipated.")
para("Baseline figures and coefficient tables use heteroskedasticity-robust HC3 inference with normal critical values. The lag structure in equation (3) contains two lags of the fundamentals and geopolitical treatments and three lags of the oil price. No HAC correction or bandwidth enters the baseline covariance. Let $A_h=\\sum_t\\widetilde Z_t\\widetilde X_t'$ and let $d_{t,h}$ be the diagonal of the projection onto the full instrument matrix, including the constant and controls. Because the models are exactly identified, this equals the leverage of the full projected-regressor design. The structural IV residual is $u_{t+h}=\\widetilde p_{t+h}−\\widetilde X_t'\\beta_h$. The HC3 influence row and covariance are defined by")
equation(r"s_{t,h}=A_h^{-1}\widetilde Z_t\frac{u_{t+h}}{1−d_{t,h}},\qquad V_h^{HC3}=\sum_t s_{t,h}s_{t,h}\prime",9)
para("The implementation uses structural residuals, not residuals from an OLS regression on fitted treatments, and includes the leverage of the controls. HC3 applies the observation-specific squared correction through the outer product in equation (9); no additional HC1 degrees-of-freedom multiplier is applied. The pointwise approximation assumes that serial covariance of the relevant instrument–error score is negligible after the specified conditioning. The lag-augmented LP literature motivates this inference strategy, but its OLS theorem is not asserted as a general result for every transformed instrument. A separate three-lag specification for every variable and conventional LP–Newey–West estimates assess sensitivity to these modeling choices.")
para("Conditional instrument strength in the joint model is evaluated with the Sanderson–Windmeijer statistic. For category $j$, the other $K−1$ endogenous regressors are instrumented by the full residualized instrument vector. The resulting conditional structural residual is regressed on all $q$ excluded instruments. If $F_{raw,j}$ is the usual homoskedastic Wald F statistic from that regression, the reported correction is")
equation(r"F_{SW,j}=F_{raw,j}\frac{q}{q−(K−1)},\qquad q=K=3",10)
para("This statistic tests whether the instrument set retains explanatory power for category $j$ after accounting for the endogenous directions spanned by the other categories. It is kept distinct from the HC3 first-stage Wald diagnostics because the classical Sanderson–Windmeijer correction is homoskedastic. The very large reported values make the substantive conclusion about conditional strength insensitive to conventional weak-instrument thresholds, but the statistic does not test exclusion.")
para("Approximate simultaneous bands use 1,999 Rademacher multiplier draws applied to the calendar-aligned HC3 influence rows. The baseline assigns an independent multiplier to each month. Six- and twelve-month blocks are supplementary path-inference sensitivity exercises, not HAC corrections to the pointwise baseline. Coverage is simultaneous over the 49 horizons of one response curve, not over every category and model. Selected-path tests jointly restrict the coefficients at horizons 0, 6, 12, 24, 36, and 48, using stacked HC3 influence rows to retain cross-horizon covariance. Cross-category equality tests use the full joint-model covariance.")

heading("5 Empirical evidence")
heading("5.1 Conditional relevance and timing",2)
para("Table 2 reports impact-horizon conditional relevance for every single-category Brent model. The instrument strongly predicts the transformed treatment for several categories, including the principal three. Relevance is less pronounced for terrorism, sanctions, and coups. The distinction matters: the fact that raw curvature is built from the underlying index does not establish that every category supplies equally informative IV variation after controls. Horizon-specific first stages are retained because the complete-case samples change with the horizon.")
rows=[]
for e,l in LABEL.items():
    r=row("first_stage",model="single",outcome="lbrent",event=e,horizon=0)
    q=R["first_stage"].query("model=='single' and outcome=='lbrent' and event==@e")
    rows.append([l,num(r.partial_r2),num(r.excluded_instrument_wald_F_hc3,1),f"{q.excluded_instrument_wald_F_hc3.min():.1f}–{q.excluded_instrument_wald_F_hc3.max():.1f}"])
table("Table 2  Single category instrument relevance for Brent",["Geopolitical index","Partial R² at impact","HC3 F at impact","F range across horizons"],rows,[2.5,1.3,1.3,1.6],"Each first stage regresses $\\ln(1+G_{j,t})$ on its raw second difference and the baseline controls, on the corresponding outcome sample. The reported one-restriction HC3 Wald statistic is numerically the robust F statistic; it uses the full first-stage design for its leverage correction. Ranges cover horizons 0–48. These are single-category relevance diagnostics, not conditional-strength statistics for the multi-endogenous-regressor model. No universal critical-value rule is imposed.")
geom=R["joint_identification"].query("model=='joint' and outcome=='lbrent'")
para(f"The joint model has rank three at every horizon. Its minimum residualized canonical correlation ranges from {geom.minimum_canonical_correlation.min():.3f} to {geom.minimum_canonical_correlation.max():.3f} for Brent. Formal Sanderson–Windmeijer conditional first-stage statistics are reported in Section 6.1. They assess whether each endogenous category retains identifying variation conditional on the other endogenous regressors and confirm that conditional relevance is strong throughout the projection horizon.")
rows=[]
for out in ["lbrent","lwti"]:
    for e in PRIMARY:
        a=row("anticipation",model="joint",outcome=out,event=e,horizon=0)
        b=row("anticipation",model="joint",outcome=out,event=e,horizon=1)
        j=row("anticipation_joint_tests",model="joint",outcome=out,event=e)
        rows.append([("Brent" if out=="lbrent" else "WTI")+" — "+LABEL[e],num(1000*a.beta),pv(a.hc3_p),num(1000*b.beta),pv(b.hc3_p),pv(j.p_value)])
table("Table 3  Oil prices before the fixed future turning point",["Outcome and category","Coefficient at t ×1000","p at t","Coefficient at t+1 ×1000","p at t+1","Joint p"],rows,[2.05,1,.6,1,.6,1.1],"Equation (8) uses $Z_{t+2}$ in both horizon regressions and origin-$t$ predetermined controls. Rows are from the joint three-category placebo. Coefficients are multiplied by 1,000 only for legibility; they are reduced-form slopes per raw instrument unit, not normalized IV responses. p-values use heteroskedasticity-robust HC3 covariance. The two-horizon test stacks calendar-aligned HC3 influence rows. Joint p tests the two displayed coefficients together using their cross-horizon covariance. Each horizon has 428 usable origins. No post-event horizon enters this table.",font=8.9)
jb=row("anticipation_joint_tests",model="joint",outcome="lbrent",event="all_three")
jw=row("anticipation_joint_tests",model="joint",outcome="lwti",event="all_three")
t=row("anticipation_joint_tests",model="single",outcome="lbrent",event="terrorism")
c=row("anticipation_joint_tests",model="single",outcome="lbrent",event="civil_war")
para(f"The principal categories do not reject the two-horizon zero restrictions in the joint placebo. The six-coefficient p-value is {pv(jb.p_value)} for Brent and {pv(jw.p_value)} for WTI. The broader evidence is less uniform: the single-category Brent civil-war test rejects (p={pv(c.p_value)}), while terrorism rejects at 10% but not 5% (p={pv(t.p_value)}). These outcomes support a category-specific assessment and are not described as universal evidence of non-anticipation. The tests concern association with future-dated curvature, not the exclusion restriction itself.")

heading("5.2 Oil price responses by event type",2)
para("Figure 3 reports all nine single-category Brent response profiles, making the breadth of the exercise and the uncertainty visible. The aggregate index averages across geopolitical developments whose implications may differ. The category-specific estimates therefore provide more economic information than an isolated aggregate response. They also show why a finding at one selected horizon should not be read as evidence that every point on a curve differs from zero. Appendix B supplies the corresponding WTI profiles.")
figure("fig03_brent_separate_irfs","Fig. 3.  Brent responses in separate geopolitical category models","Each panel estimates equation (2) using one endogenous $\\ln(1+G_{j,t})$ regressor instrumented by its raw second difference. Panels are aggregate AI-GPR, military conflict, diplomatic tension, terrorism, civil war, nuclear threat, coup, sanctions, and other events. Responses are normalized to a 10% increase in one plus the category index and measured in log points ×100. Dark and light shaded areas are 90% and 95% heteroskedasticity-robust HC3 pointwise intervals. The baseline includes two lags of activity, oil-production growth, and the included geopolitical regressor, plus three oil-price lags. Horizons are monthly; samples are complete-case and horizon-specific. The exploratory categories require the relevance and timing qualifications in Section 5.1.")
para("The principal joint model in Figure 4 separates co-moving military, diplomatic, and nuclear variation. Table 4 reports the impact, 12-month, and 24-month coefficients, the two covariance estimates, and exact percentage equivalents. These are effects of variation in the corresponding index conditional on the other two current instrumented categories, rather than separate univariate responses pasted together. The same controls and transformation are applied to Brent and WTI, making their comparison internally consistent.")
rows=[]
for out in ["lbrent","lwti"]:
    for e in PRIMARY:
        for h in [0,12,24]:
            r=beta(out,e,h)
            rows.append([("Brent" if out=="lbrent" else "WTI")+" — "+LABEL[e],str(h),num(r.beta),num(r.hc3_se),num(r.hc1_se),num(pct(r.beta),2)])
table("Table 4  Joint geopolitical responses at selected horizons",["Outcome and category","h","IV slope","HC3 SE","HC1 SE","Effect %"],rows,[2.5,.35,.9,.9,.9,1.15],"The joint model includes military conflict, diplomatic tension, and nuclear threat simultaneously, instrumented by their three raw second differences. IV slopes are coefficients on $\\ln(1+G)$. Effect % equals $100[\\exp(\\beta_h\\ln(1.10))−1]$, a 10% increase in one plus that index with the other current categories held fixed in the model. HC3 uses leverage-adjusted structural IV residuals, with no HAC correction or bandwidth. HC1 is a sensitivity comparison. N=428, 424, and 412 at h=0, 12, and 24, respectively. No significance stars are used; uncertainty is reported directly.",font=8.9)
figure("fig04_joint_irfs","Fig. 4.  Joint geopolitical responses of Brent and WTI","Rows correspond to military conflict, diplomatic tension, and nuclear threat; the left column is Brent and the right column WTI. Equation (6) instruments all three current category regressors jointly with their raw second differences. Responses are measured in log points ×100 for a 10% increase in one plus the corresponding index. Dark and light areas are 90% and 95% heteroskedasticity-robust HC3 pointwise intervals. The same baseline lag augmentation is used in each panel. The limits of the vertical axes are matched within each row, not across different event types. These are pointwise intervals, not simultaneous path bands.",width=6.4)
m0=beta("lbrent","military_conflict",0); m24=beta("lbrent","military_conflict",24)
para(f"Military conflict raises the estimated Brent price on impact: the joint coefficient is {num(m0.beta)}, corresponding to {pct(m0.beta):.2f}% for the stated normalization. At 24 months it is {num(m24.beta)} and its confidence interval includes zero. The short-run positive response is compatible with disruption concerns or precautionary demand, but it does not establish which mechanism dominates. Nor does an impact rejection justify a claim that the military response is nonzero over the entire four-year horizon.")
para(f"Diplomatic tension exhibits a negative medium-run response. At 24 months the joint Brent slope is {num(dip.beta)}, with HC3 standard error {num(dip.hc3_se)} and HC1 sensitivity standard error {num(dip.hc1_se)}. Its exact normalized price response is {pct(dip.beta):.2f}%. A deterioration in diplomatic conditions can depress expectations of trade and global activity, reducing expected oil demand. The sign is consistent with that interpretation, but the specification does not estimate expectations or trade quantities as separate mediators. The negative price response is a reduced description of the total instrumented geopolitical effect.")
para(f"The nuclear-threat response is positive at the same horizon: the joint Brent coefficient is {num(nuc.beta)}, with HC3 standard error {num(nuc.hc3_se)} and HC1 sensitivity standard error {num(nuc.hc1_se)}, implying {pct(nuc.beta):.2f}%. A threat with a low immediate probability of realized disruption may still affect precautionary behavior and the valuation of adverse future states. The joint estimate is larger than the separate-category nuclear estimate at this horizon, illustrating why accounting for co-moving diplomatic and military conditions changes the conditional estimand. This full-sample result must also be read alongside the period sensitivity below.")
eq=R["cross_category_equality"]
eq=eq[(eq.outcome=="lbrent")&(eq.test=="all_three_equal")&(eq.covariance=="HC3")&(eq.scope=="pointwise")]
pe={str(r.horizon):r.p_value for _,r in eq.iterrows()}
para(f"A formal equality test in the joint Brent model does not reject at impact (HC3 p={pv(pe['0'])}), but rejects at 12 months (p={pv(pe['12'])}) and 24 months (p={pv(pe['24'])}). Thus the strongest evidence of cross-category heterogeneity is in the medium-run response. The tested null is equality of slopes under an equal $\\ln(1+G)$ normalization; it is not equality of effects of three historically equivalent events. The selected-path equality test also rejects, using the cross-horizon and cross-category covariance rather than treating the coefficient estimates as independent.")

heading("6 Robustness and limits of interpretation")
heading("6.1 Alternative inference and conditional instrument strength",2)
para("Table 5 compares the lag-augmented HC3 baseline with a conventional projection using two price lags and a six-month Newey–West correction. Each pair is re-estimated on exactly the same observations. Both the dynamic specification and covariance estimator differ, so the conventional coefficient is reported separately rather than attaching its standard error to the augmented estimate. Additional conventional-model bandwidths of 4, 12, 24, and 48 months are retained in the supplementary results. HC1 is also reported as a finite-sample covariance sensitivity for the unchanged augmented model.")
rows=[]
for e in PRIMARY:
    for h in [0,12,24]:
        q=row("inference_comparison",model="joint",outcome="lbrent",event=e,horizon=h)
        rows.append([LABEL[e],str(h),num(q.augmented_beta),num(q.augmented_hc3_se),num(q.conventional_beta),num(q.conventional_hac6_se)])
table("Table 5  Lag augmentation and conventional inference for Brent",["Category","h","Augmented slope","HC3 SE","Conventional slope","NW6 SE"],rows,[1.9,.3,1.15,1,1.3,1.1],"Both models are IV local projections with the same instruments and matched observations. The augmented equation uses three price lags and HC3; the conventional equation uses two price lags and six-month Bartlett Newey–West standard errors. The other lag blocks contain two lags in both regressions. Each standard error belongs to the coefficient displayed immediately to its left. These are alternative estimators, not two corrections applied to one baseline.",font=8.7)
sw=R["sw_conditional_first_stage"].query("outcome=='lbrent'")
rows=[]
for e in PRIMARY:
    sb=sw.query("event==@e").sw_conditional_F
    swti=R["sw_conditional_first_stage"].query("outcome=='lwti' and event==@e").sw_conditional_F
    rows.append([LABEL[e],f"{sb.min():.1f}–{sb.max():.1f}",f"{swti.min():.1f}–{swti.max():.1f}"])
table("Table 6  Sanderson Windmeijer conditional first stage statistics",["Category","Brent horizons 0–48","WTI horizons 0–48"],rows,[2.4,2.15,2.15],"The statistic tests the conditional first stage for each endogenous category in the joint three-regressor model. Each cell reports the minimum and maximum across the 49 horizon-specific complete-case samples. The classical Sanderson–Windmeijer correction is homoskedastic and is reported separately from the HC3 single-category relevance statistics in Table 2. All values are far above conventional weak-instrument ranges. The diagnostic establishes conditional relevance; it does not test exclusion.",font=9.1)
sw_ranges = {}
for outcome in ["lbrent", "lwti"]:
    for event in PRIMARY:
        values = R["sw_conditional_first_stage"].query("outcome==@outcome and event==@event").sw_conditional_F
        sw_ranges[(outcome, event)] = (values.min(), values.max())
para("Conditional relevance is uniformly strong in the joint specification. Across the 49 Brent horizons, the Sanderson–Windmeijer statistic ranges from "
     f"{sw_ranges[('lbrent','military_conflict')][0]:.1f} to {sw_ranges[('lbrent','military_conflict')][1]:.1f} for military conflict, from "
     f"{sw_ranges[('lbrent','diplomatic_tension')][0]:.1f} to {sw_ranges[('lbrent','diplomatic_tension')][1]:.1f} for diplomatic tension, and from "
     f"{sw_ranges[('lbrent','nuclear_threat')][0]:.1f} to {sw_ranges[('lbrent','nuclear_threat')][1]:.1f} for nuclear threat. The corresponding WTI minima are "
     f"{sw_ranges[('lwti','military_conflict')][0]:.1f}, {sw_ranges[('lwti','diplomatic_tension')][0]:.1f}, and {sw_ranges[('lwti','nuclear_threat')][0]:.1f}. "
     "These values establish exceptionally strong conditional relevance, although they do not bear on exclusion.")
sim=R["simultaneous_bands"]
sd=sim.query("model=='joint' and outcome=='lbrent' and event=='diplomatic_tension' and block_length==1")
sn=sim.query("model=='joint' and outcome=='lbrent' and event=='nuclear_threat' and block_length==1")
def significant_h(q): return ", ".join(str(int(h)) for h in q.loc[(q.sim95_low>0)|(q.sim95_high<0),"horizon"]) or "none"
para("Simultaneous bands address the uncertainty associated with examining the entire response curve. With independent monthly multipliers, the 95% joint Brent diplomatic band contains zero at every horizon, despite pointwise rejections in the medium run. The nuclear band excludes zero at months "+significant_h(sn)+". Thus evidence for a nonzero diplomatic response is weaker when inference is adjusted for inspecting the entire curve. Six- and twelve-month blocks assess sensitivity of the simultaneous calculation to dependence across months. These fixed-design influence approximations are not resampling estimates of geopolitical events and do not provide global multiplicity control over all models.")

heading("6.2 Episodes and estimation periods",2)
para("The event-exclusion exercise removes origin months within one month of twelve major geopolitical episodes, one episode at a time, with all lags and leads constructed beforehand. The episodes include the Gulf conflicts, the 1998 South Asian nuclear tests, September 11, North Korean escalations, the Libyan uprising, the Russian invasion of Ukraine, and the October 2023 outbreak of war between Israel and Hamas. These are checks on the contribution of origin-month windows, not estimates on a sample purged of every future outcome influenced by those events. In particular, a retained earlier origin may still have a long-horizon outcome falling in an excluded event month.")
for e,h in [("diplomatic_tension",24),("nuclear_threat",24),("military_conflict",0)]:
    q=R["event_drop"].query("outcome=='lbrent' and event==@e and horizon==@h")
    para(f"For {LABEL[e].lower()} at horizon {h}, the separate-category Brent coefficient ranges from {num(q.beta.min())} to {num(q.beta.max())} across the twelve origin-window exclusions. The baseline coefficient is {num(beta('lbrent',e,h,'single').beta)}. The range describes sensitivity to these particular windows; it does not identify which geopolitical episode supplied an exogenous intervention, and it is not a confidence interval.")
lev=R["leverage"]
lev=lev[(lev.outcome=="lbrent")&(lev.horizon==0)&(lev.contribution_type=="partial")&(lev["rank"]==1)]
if len(lev)==0: lev=R["leverage"].query("outcome=='lbrent' and horizon==0 and contribution_type=='partialled' and rank==1")
para("An additional leverage diagnostic ranks the absolute contributions to the residualized instrument–regressor covariance. It distinguishes the visual size of a raw spike from its contribution after conditioning on the controls. The estimates use many months, but some categories are more concentrated in a few episodes. Concentration is relevant to interpretation because a full-sample coefficient can primarily reflect the kinds of events receiving the greatest effective IV weight, rather than a representative mix of geopolitical developments.")
rows=[]
for e in PRIMARY:
    for spec in ["pre_2012","pre_2019"]:
        r=row("origin_splits",outcome="lbrent",event=e,horizon=24,specification=spec)
        rows.append([LABEL[e],spec.replace("_"," "),str(int(r.n)),num(r.beta),num(r.hc3_se)])
table("Table 7  Historical stability of the separate Brent response at 24 months",["Category","Origin period","N","IV slope","HC3 SE"],rows,[2.15,1.4,.6,1.2,1.35],"Splits refer to regression-origin months. Pre 2012 ends in December 2011 and pre 2019 ends in December 2018. Outcome dates extend beyond the origin-period cutoff. The two historical samples are nested and therefore are not independent. Each row uses the baseline lag structure, single-category instrumentation, and HC3 standard errors. The exercise is descriptive and does not impose or estimate a structural-break date.")
nuc12=row("origin_splits",outcome="lbrent",event="nuclear_threat",horizon=24,specification="pre_2012")
nuc19=row("origin_splits",outcome="lbrent",event="nuclear_threat",horizon=24,specification="pre_2019")
para(f"The 24-month nuclear-threat response is stable in the two long historical samples. Its separate Brent coefficient is {num(nuc12.beta)} with an HC3 standard error of {num(nuc12.hc3_se)} through December 2011 and {num(nuc19.beta)} with an HC3 standard error of {num(nuc19.hc3_se)} through December 2018. Both estimates are positive and statistically different from zero at conventional levels. The negative diplomatic coefficient likewise retains its sign in the displayed historical samples.")

heading("6.3 Specification sensitivity",2)
para("Lag sensitivity is examined on matched observations. A two-price-lag equation removes the baseline augmentation, a four-price-lag equation adds another price lag, and a third alternative uses three lags of every variable in the control system. The latter directly checks whether extending the activity, production-growth, and geopolitical lag blocks changes the estimates. Table B1 reports these comparisons and the contemporaneous-production control. The production extension changes the conditioning set and may remove part of geopolitical transmission; it is not an externally identified oil-supply shock.")
para("The log-curvature contrast replaces the raw second difference by $\\Delta^2x_{j,t}$. Since $x_{j,t−1}$ and $x_{j,t−2}$ are already controls, residualizing this alternative instrument gives exactly the residualized current regressor. Its IV coefficient consequently equals the ordinary-least-squares coefficient on the same sample. This algebraic result explains why the contrast must not be interpreted as a second independent instrument that corroborates exclusion. It documents the consequences of the instrument's transformation and reinforces the need to specify raw-index curvature in the baseline.")
para("Several boundaries remain. News-based measurement may include variation unrelated to the economically relevant state. A global event index aggregates heterogeneous locations and severities. The macro controls do not exhaust information available to market participants. Identification assumes that the instrument-induced variation represented by the geopolitical state captures the relevant pathway into prices; the diagnostic tests cannot prove this condition. These limits qualify the scope of the causal interpretation without making expectations, risk pricing, or other consequences of geopolitics illegitimate transmission channels.")

heading("7 Energy security policy implications")
heading("7.1 Event specific monitoring",2)
para("The first policy implication is that an aggregate geopolitical-risk index should not be used as a sufficient statistic for oil-market stress. The estimated signs, timing, and persistence differ across military conflict, diplomatic tension, and nuclear threat. Energy ministries, emergency-stock agencies, and market-monitoring units can therefore obtain a more informative signal by reporting the event composition of geopolitical risk alongside its aggregate level. A dashboard should show whether a monthly increase is driven by threats to physical security, armed conflict, diplomatic confrontation, sanctions, or another category, and whether the change represents a persistent level shift or a sharp turning point.")
para("This recommendation concerns diagnosis, not automatic action. The category indices measure news coverage and severity scores; they are not direct measures of disrupted barrels, spare capacity, or transport availability. A large turning point should trigger closer examination of physical-market evidence rather than a predetermined policy response. The dated episodes in Appendix A illustrate how recognizable events enter the curvature measure, while the event-window tests show that the main estimates are not mechanically attributable to one selected crisis.")

heading("7.2 Contingency planning and emergency stocks",2)
para("The second implication concerns the sequencing of contingency decisions. The positive nuclear-threat response is consistent with markets pricing disruption risk or precautionary demand, whereas the negative medium-run diplomatic response is consistent with weaker expected global activity. These mechanisms imply different policy problems. A prospective availability shortfall calls for operational readiness, coordination with other stockholding countries, and verification of release capacity. A demand-driven price decline does not provide the same rationale for an emergency release and may instead raise questions about revenue exposure, investment, and the resilience of producing economies.")
para("The estimates do not identify the effect of releasing strategic petroleum reserves and cannot determine an optimal release rule. They support a two-stage decision process. In the first stage, event-specific geopolitical information determines which contingency scenarios deserve attention. In the second, intervention depends on corroborating indicators: observed and expected production losses, export restrictions, shipping and insurance conditions, commercial inventories, spare capacity, refinery constraints, and futures spreads. This safeguard prevents the geopolitical index from becoming a mechanical trigger while still using it as timely information about the nature of the risk.")

heading("7.3 Price scenarios inflation risk and communication",2)
para("The response horizon also matters for planning. Military conflict produces a positive impact estimate but no precise medium-run response, whereas diplomatic and nuclear categories diverge more clearly around the two-year horizon. Short-run emergency planning and medium-run macroeconomic scenarios should therefore not apply one common geopolitical multiplier. Agencies can construct category-specific paths, preserve the uncertainty bands, and update the scenario when physical data reveal whether supply, demand, or precautionary forces dominate.")
para("The results are also relevant to communication between energy authorities, fiscal agencies, and central banks. A geopolitical escalation is not necessarily an inflationary oil-supply shock. Communicating the event composition and the supporting physical evidence can reduce the risk of treating a demand-related decline and a disruption-related increase as equivalent. Consumer-price pass-through, welfare effects, and monetary-policy responses are outside the estimated system and require separate models. The contribution here is to improve the oil-price input used in such assessments.")

heading("7.4 Limits to policy use",2)
para("Policy application should retain four qualifications. First, the estimates are global historical responses local to the variation selected by the instruments, not forecasts for a named future crisis. Second, a 10% increase in one plus an index is a common statistical normalization, not an assertion that different events have equal physical severity. Third, the positive nuclear response is stable in the long samples ending before 2012 and before 2019, while simultaneous inference is less decisive for some diplomatic horizons than pointwise inference. Fourth, inventories, shipping disruptions, and policy interventions are not separately identified as transmission mechanisms. The appropriate use is disciplined scenario formation and monitoring, not a deterministic intervention rule.")

heading("8 Conclusion")
para("Energy policy should not begin from the premise that every geopolitical escalation is the same adverse oil-supply shock. Event-specific turning points reveal dynamic differences concealed by the aggregate risk measure. In the full-sample joint model, diplomatic tension is followed by lower real oil prices in the medium run, while nuclear threat has the opposite sign. Military conflict has a positive impact response but a less precise medium-run profile. Formal equality tests support heterogeneity at medium horizons, and the comparison of Brent and WTI shows that the finding is not confined to one benchmark.")
para("The practical contribution is an event-specific diagnostic for energy-security assessment. Geopolitical categories can identify which supply-disruption, demand, and precautionary scenarios deserve closer attention; physical-market indicators must then determine whether emergency intervention is justified. This approach supports differentiated reserve readiness, horizon-specific stress testing, and clearer communication of oil-price and inflation risks without converting a news index into an automatic policy rule.")
para("The causal interpretation rests on the economic argument that abrupt changes in geopolitical trajectories provide variation less plausibly driven by routine monthly economic feedback, and that this variation affects prices through the geopolitical state and its consequences. The published turning-point strategy supplies the identification framework; the present evidence shows its value for distinguishing energy-market responses across geopolitical event types. The principal timing tests do not reject, the core patterns survive event exclusions, and the nuclear response remains positive in the long samples ending before 2012 and before 2019. The results support a transparent and conditional use of geopolitical information in energy policy: disaggregate the event, assess its likely mechanism, corroborate it with physical evidence, and preserve uncertainty in the resulting scenario.")

heading("References")
refs=[
"Baumeister, C., Hamilton, J.D., 2019. Structural interpretation of vector autoregressions with incomplete identification: Revisiting the role of oil supply and demand shocks. American Economic Review 109(5), 1873–1910. https://doi.org/10.1257/aer.20151569.",
"Caldara, D., Iacoviello, M., 2022. Measuring geopolitical risk. American Economic Review 112(4), 1194–1225. https://doi.org/10.1257/aer.20191823.",
"Cherp, A., Jewell, J., 2014. The concept of energy security: Beyond the four As. Energy Policy 75, 415–421. https://doi.org/10.1016/j.enpol.2014.09.005.",
"Iacoviello, M., Tong, J., 2026. The AI-GPR Index: Measuring Geopolitical Risk using Artificial Intelligence. Working paper, Federal Reserve Board of Governors, September 13. https://www.matteoiacoviello.com/research_files/AI_GPR_PAPER.pdf. Data: https://www.matteoiacoviello.com/ai_gpr.html.",
"Jordà, Ò., 2005. Estimation and inference of impulse responses by local projections. American Economic Review 95(1), 161–182. https://doi.org/10.1257/0002828053828518.",
"Kilian, L., 2009. Not all oil price shocks are alike: Disentangling demand and supply shocks in the crude oil market. American Economic Review 99(3), 1053–1069. https://doi.org/10.1257/aer.99.3.1053.",
"Kilian, L., Murphy, D.P., 2014. The role of inventories and speculative trading in the global market for crude oil. Journal of Applied Econometrics 29(3), 454–478. https://doi.org/10.1002/jae.2322.",
"Li, D., Plagborg-Møller, M., Wolf, C.K., 2024. Local projections vs. VARs: Lessons from thousands of DGPs. Journal of Econometrics 244(2), 105722. https://doi.org/10.1016/j.jeconom.2024.105722.",
"Mignon, V., Saadaoui, J., 2024. How do political tensions and geopolitical risks impact oil prices? Energy Economics 129, 107219. https://doi.org/10.1016/j.eneco.2023.107219.",
"Mignon, V., Saadaoui, J., 2025. Asymmetries in the oil market: accounting for the growing role of China through quantile regressions. Macroeconomic Dynamics 29, e38. https://doi.org/10.1017/S1365100524000312.",
"Montiel Olea, J.L., Plagborg-Møller, M., 2019. Simultaneous confidence bands: Theory, implementation, and an application to SVARs. Journal of Applied Econometrics 34(1), 1–17. https://doi.org/10.1002/jae.2656.",
"Montiel Olea, J.L., Plagborg-Møller, M., 2021. Local projection inference is simpler and more robust than you think. Econometrica 89(4), 1789–1823. https://doi.org/10.3982/ECTA18756.",
"Nakamura, E., Steinsson, J., 2018. Identification in macroeconomics. Journal of Economic Perspectives 32(3), 59–86. https://doi.org/10.1257/jep.32.3.59.",
"Plagborg-Møller, M., Wolf, C.K., 2021. Local projections and VARs estimate the same impulse responses. Econometrica 89(2), 955–980. https://doi.org/10.3982/ECTA17813.",
"Saadaoui, J., 2026. Geopolitical turning points and macroeconomic volatility: A bilateral identification strategy. Journal of Comparative Economics 54, 804–818. https://doi.org/10.1016/j.jce.2026.03.010.",
"Stock, J.H., Watson, M.W., 2018. Identification and estimation of dynamic causal effects in macroeconomics using external instruments. Economic Journal 128(610), 917–948. https://doi.org/10.1111/ecoj.12593.",
"Verduzco-Bustos, G., Zanetti, F., 2026. The effects of geopolitical oil price shocks. CEPR Discussion Paper 21378. https://cepr.org/publications/dp21378.",
"Winzer, C., 2012. Conceptualizing energy security. Energy Policy 46, 36–48. https://doi.org/10.1016/j.enpol.2012.02.067.",
]
refs += ['Alquist, R., Kilian, L., 2010. What do we learn from the price of crude oil futures? Journal of Applied Econometrics 25(4), 539–573. https://doi.org/10.1002/jae.1159.', 'Anderson, T.W., Rubin, H., 1949. Estimation of the parameters of a single equation in a complete system of stochastic equations. Annals of Mathematical Statistics 20(1), 46–63. https://doi.org/10.1214/aoms/1177730090.', 'Andrews, I., Stock, J.H., Sun, L., 2019. Weak instruments in instrumental variables regression: Theory and practice. Annual Review of Economics 11, 727–753. https://doi.org/10.1146/annurev-economics-080218-025643.', 'Angrist, J.D., Imbens, G.W., 1995. Two-stage least squares estimation of average causal effects in models with variable treatment intensity. Journal of the American Statistical Association 90(430), 431–442. https://doi.org/10.1080/01621459.1995.10476535.', 'Baker, S.R., Bloom, N., Davis, S.J., 2016. Measuring economic policy uncertainty. Quarterly Journal of Economics 131(4), 1593–1636. https://doi.org/10.1093/qje/qjw024.', 'Barnichon, R., Brownlees, C., 2019. Impulse response estimation by smooth local projections. Review of Economics and Statistics 101(3), 522–530. https://doi.org/10.1162/rest_a_00778.', 'Baumeister, C., Kilian, L., 2016. Forty years of oil price fluctuations: Why the price of oil may still surprise us. Journal of Economic Perspectives 30(1), 139–160. https://doi.org/10.1257/jep.30.1.139.', 'Bloom, N., 2009. The impact of uncertainty shocks. Econometrica 77(3), 623–685. https://doi.org/10.3982/ECTA6248.', 'Caldara, D., Cavallo, M., Iacoviello, M., 2019. Oil price elasticities and oil price fluctuations. Journal of Monetary Economics 103, 1–20. https://doi.org/10.1016/j.jmoneco.2018.08.004.', 'Caldara, D., Iacoviello, M., Molligo, P., Prestipino, A., Raffo, A., 2020. The economic effects of trade policy uncertainty. Journal of Monetary Economics 109, 38–59. https://doi.org/10.1016/j.jmoneco.2019.11.002.', 'Handley, K., Limão, N., 2017. Policy uncertainty, trade, and welfare: Theory and evidence for China and the United States. American Economic Review 107(9), 2731–2783. https://doi.org/10.1257/aer.20141419.', 'Inoue, A., Rossi, B., Wang, Y., 2024. Local projections in unstable environments. Journal of Econometrics 244(2), 105726. https://doi.org/10.1016/j.jeconom.2024.105726.', 'Jordà, Ò., Taylor, A.M., 2025. Local projections. Journal of Economic Literature 63(1), 59–110. https://doi.org/10.1257/jel.20241521.', 'MacKinnon, J.G., White, H., 1985. Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. Journal of Econometrics 29(3), 305–325. https://doi.org/10.1016/0304-4076(85)90158-7.', 'Montiel Olea, J.L., Pflueger, C., 2013. A robust test for weak instruments. Journal of Business and Economic Statistics 31(3), 358–369. https://doi.org/10.1080/00401706.2013.806694.', 'Newey, W.K., West, K.D., 1987. A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica 55(3), 703–708. https://doi.org/10.2307/1913610.', 'Sanderson, E., Windmeijer, F., 2016. A weak instrument F-test in linear IV models with multiple endogenous variables. Journal of Econometrics 190(2), 212–221. https://doi.org/10.1016/j.jeconom.2015.06.004.', 'Staiger, D., Stock, J.H., 1997. Instrumental variables regression with weak instruments. Econometrica 65(3), 557–586. https://doi.org/10.2307/2171753.', 'White, H., 1980. A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. Econometrica 48(4), 817–838. https://doi.org/10.2307/1912934.']
refs = [s for s in refs if not s.startswith(("Anderson, T.W., Rubin", "Andrews, I., Stock", "Montiel Olea, J.L., Pflueger, C., 2013"))]
for s in sorted(refs):
    p=para(s); p.paragraph_format.first_line_indent=Inches(-.2); p.paragraph_format.left_indent=Inches(.2)
    p.paragraph_format.line_spacing=1; p.paragraph_format.space_after=Pt(6); p.paragraph_format.keep_together=True
    for r in p.runs: r.font.size=Pt(10)

doc.add_page_break()
heading("Appendix A Data sources and construction")
para("The estimation snapshot is monthly, uses a complete calendar, and contains no imputed observations. AI-GPR is from Iacoviello and Tong's official monthly event-type dataset, vintage through August 31, 2026. The aggregate and eight event-category columns retain the source units. The treatment is computed as the natural logarithm of one plus each index. The instrument is computed from the untransformed index. Source and transformed data are supplied as both CSV and XLSX, with missing observations left blank and a variable dictionary recording these definitions.")
para("Nominal WTI is the U.S. Energy Information Administration Cushing, Oklahoma spot price, monthly average; nominal Brent is the EIA Europe Brent spot price, monthly average. Both complete EIA histories were downloaded on September 19, 2026. Each is deflated by the seasonally adjusted all-items CPI series CUSR0000SA0 from the Bureau of Labor Statistics bulk-data file, also downloaded on September 19, 2026. The resulting real-price levels are logged. Official source pages are https://www.eia.gov/dnav/pet/pet_pri_spt_s1_m.htm and https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems. The logarithmic price normalization affects intercepts but not estimated responses when a constant is included.")
para("World industrial production is the Baumeister–Hamilton index for OECD economies and six major non-member economies, available from https://sites.google.com/site/cjsbaumeister/datasets. Global crude-oil production is EIA international series 57-1-WORL-TBPD.M, obtained through DBnomics at https://db.nomics.world/EIA/INTL/57-1-WORL-TBPD.M. The first difference of the log production level is the production-growth control; it is not divided by 100 or annualized. Monthly logs of industrial production enter the control vector in levels.")
para("Every macroeconomic variable is constructed from one complete current-source history. No historical segment is retained from an earlier research vintage; no later segment is spliced or rebased at a join date. The package includes the full provider files and a build script that recreates the monthly merge directly. CUSR0000SA0 reports no October 2025 observation in the downloaded BLS file. Its level is therefore filled by log-linear interpolation between September and November 2025, yielding approximately 324.654. The corresponding nominal Brent and WTI observations are taken directly from the EIA. This is the only interpolated input; other unavailable provider values remain missing.")
rows=[]
for _,r in R["endpoints"].iterrows():
    if r.variable in ("lwti","lbrent","lwip","lgop","x_gpr_ai"):
        lab={"lwti":"Real WTI price","lbrent":"Real Brent price","lwip":"World industrial production","lgop":"Global oil production","x_gpr_ai":"AI-GPR and event categories"}[r.variable]
        rows.append([lab,str(r.first_available)[:7],str(r.last_available)[:7],str(int(r.nonmissing)),str(int(r.missing))])
table("Table A1  Observation availability",["Series","First month","Last month","Observed","Missing"],rows,[2.7,1,1,.95,1.05],"Availability is evaluated over the 440-month calendar January 1990–August 2026, before transformations. Missing includes months after a series' source endpoint. Real Brent and WTI are complete: the unavailable October 2025 CPI deflator is log-linearly interpolated, while the nominal oil prices remain observed. Two instrument differences and the lagged controls further reduce admissible regression origins.")
event_specs = [
    ("1990-08-01", "August 1990", "Iraq invades Kuwait", "military_conflict", "1990-07 to 1990-09"),
    ("1991-01-01", "January 1991", "Gulf War air campaign", "military_conflict", "1990-12 to 1991-02"),
    ("1998-05-01", "May 1998", "Indian and Pakistani nuclear tests", "nuclear_threat", "1998-04 to 1998-06"),
    ("2001-09-01", "September 2001", "September 11 attacks", "terrorism", "2001-08 to 2001-10"),
    ("2003-03-01", "March 2003", "Invasion of Iraq", "military_conflict", "2003-02 to 2003-04"),
    ("2006-10-01", "October 2006", "North Korean nuclear test", "nuclear_threat", "2006-09 to 2006-11"),
    ("2011-02-01", "February 2011", "Libyan uprising", "civil_war", "2011-01 to 2011-03"),
    ("2013-07-01", "July 2013", "Egyptian military takeover", "coup", "2013-06 to 2013-08"),
    ("2015-11-01", "November 2015", "Paris attacks", "terrorism", "2015-10 to 2015-12"),
    ("2017-07-01", "July 2017", "North Korean missile escalation", "nuclear_threat", "2017-06 to 2017-08"),
    ("2022-02-01", "February 2022", "Russian invasion of Ukraine", "military_conflict", "2022-01 to 2022-03"),
    ("2023-10-01", "October 2023", "Israel-Hamas war outbreak", "military_conflict", "2023-09 to 2023-11"),
]
event_data = pd.read_csv(ROOT / "data" / "monthly_data.csv", parse_dates=["date"]).set_index("date")
event_curvature = event_data[list(LABEL)].diff().diff()
event_rows = []
for date, month, episode, category, window in event_specs:
    dated = pd.Timestamp(date)
    values = event_curvature.loc[dated]
    history = event_curvature.loc[:dated, category].dropna().abs()
    percentile = 100 * (history <= abs(values[category])).mean()
    event_rows.append([
        month, episode, LABEL[category], num(values[category], 2),
        "Positive" if values[category] > 0 else "Negative",
        f"{percentile:.1f}", window,
    ])
table("Table A2  Dated geopolitical episodes used in the event-window analysis",
      ["Dated month", "Geopolitical episode", "Illustrative category", "Raw curvature Z", "Sign", "Historical |Z| percentile", "Excluded window"],
      event_rows, [.9, 1.75, 1.15, .7, .65, .85, .95],
      "Raw curvature is the signed turning-point value for the event type most directly associated with the dated episode: $Z_{j,t}=\\Delta^2G_{j,t}$. Positive values indicate upward curvature toward higher measured category risk; negative values indicate downward curvature. The historical percentile ranks $|Z_{j,t}|$ within the absolute category-specific curvature observations available from January 1990 through the dated month, inclusive. It therefore measures the contemporaneous extremity of the turning point without using future data. The category assignment is illustrative rather than exclusive because several event types can move in the same month. The event-exclusion exercise removes the dated month and its adjacent origin months, one episode at a time.", font=8.2)
para("Table A2 restores the quantitative correspondence between recognizable geopolitical episodes and the instrument. The signed value identifies the direction of the change in trajectory, while the expanding historical percentile shows how unusual its magnitude was when the episode occurred. These are descriptive realizations of the rule-based instrument, not hand-coded shocks or proof that each episode is exogenous. The dates neither define the instrument nor select the baseline observations. Complete event-window estimates by outcome, category, horizon, and episode are reported in the replication file results/event_drop.csv.")
para("Software constructs leads and lags before trimming the sample, uses stable partialling-out calculations for the exactly identified IV estimator, and retains calendar-aligned score arrays for covariance calculations. Reported tables are generated from the exported numerical estimates rather than manually entered coefficients. Tests independently compare the estimator and covariance matrices with alternative calculations and verify the fixed two-month lead, lag patterns, missing-data handling, and shock normalization. Those checks establish implementation consistency; the economic identification conditions remain the assumptions stated in Section 4.")

heading("Appendix B Supplementary specification and price benchmark")
rows=[]
for e,h in [("military_conflict",0),("diplomatic_tension",24),("nuclear_threat",24)]:
    for data,spec,label in [("lag_sensitivity","non_augmented_2_price_lags","2 price lags"),("lag_sensitivity","extra_4_price_lags","4 price lags"),("lag_sensitivity","three_lags_all_variables","3 lags all variables"),("production_control","current_production_growth_control","Current production")]:
        r=row(data,outcome="lbrent",event=e,horizon=h,specification=spec)
        rows.append([LABEL[e]+f" h={h}",label,num(r.matched_baseline_beta),num(r.beta),num(r.hc3_se),str(int(r.n))])
table("Table B1  Matched sample specification sensitivity for Brent",["Category and horizon","Alternative","Matched baseline","Alternative slope","HC3 SE","N"],rows,[2.0,1.35,1,1,.8,.55],"Each alternative is compared with the single-category baseline re-estimated on exactly the same observations. Baseline coefficients use three price lags. Alternatives use two or four price lags, three lags of every variable, or contemporaneous oil-production growth. Apart from the stated change, controls are unchanged. Standard errors use HC3 for the displayed alternative. The production-control exercise conditions on a potentially endogenous mediator and is not an independently identified oil-supply-shock control.",font=8.8)
figure("fig03_wti_separate_irfs","Fig. B1.  WTI responses in separate geopolitical category models","The specification, panel order, normalization, and interval convention are identical to Fig. 3, with log real WTI replacing log real Brent. Responses are log points ×100 for a 10% increase in one plus the corresponding geopolitical index. The treatment uses the natural logarithm of one plus the index; the instrument is the second difference of its raw level. Dark and light areas are 90% and 95% heteroskedasticity-robust HC3 pointwise intervals.")

path=OUT/"Geopolitical_Turning_Points_and_Oil_Prices_V3.docx"
for root in (doc.styles._element, doc._element):
    for border in list(root.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
    for fonts in root.iter(qn("w:rFonts")):
        for attr in list(fonts.attrib):
            if "theme" in attr.lower(): del fonts.attrib[attr]
        for key in ("ascii", "hAnsi", "cs", "eastAsia"):
            fonts.set(qn("w:"+key), FONT)
doc.save(path)
print(path)
print("Approximate prose words:", sum(len(p.text.split()) for p in doc.paragraphs))
