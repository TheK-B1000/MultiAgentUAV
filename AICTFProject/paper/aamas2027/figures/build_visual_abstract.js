/**
 * Low-text visual abstract — shapes first, words second.
 *   node build_visual_abstract.js
 */
const pptxgen = require("pptxgenjs");
const path = require("path");

const T = {
  bg: "F4F8FA",
  ink: "0B2C3A",
  ink2: "1A4A5C",
  ice: "CADCFC",
  coral: "E07A5F",
  gold: "D4A017",
  teal: "0E6B7A",
  sea: "2A9D8F",
  slate: "5C7A86",
  white: "FFFFFF",
};

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
  pres.title = "Visual abstract (low text)";
  const slide = pres.addSlide();
  slide.background = { color: T.bg };

  // Header
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 13.3, h: 1.15,
    fill: { color: T.ink },
  });
  slide.addText("Opponent-conditioned specialization under policy sharing", {
    x: 0.5, y: 0.22, w: 12.3, h: 0.45,
    fontSize: 26, bold: true, color: T.white, fontFace: "Calibri",
    margin: 0, isTextBox: true,
  });
  slide.addText("regimes  →  teachers  →  shared student  →  crossover", {
    x: 0.5, y: 0.7, w: 12.3, h: 0.3,
    fontSize: 14, color: T.ice, fontFace: "Calibri",
    margin: 0, isTextBox: true,
  });

  const panels = [
    { x: 0.4, accent: T.coral, label: "1  Regimes" },
    { x: 3.55, accent: T.teal, label: "2  Teachers" },
    { x: 6.7, accent: T.ink2, label: "3  Distill" },
    { x: 9.85, accent: T.sea, label: "4  Test" },
  ];

  for (const p of panels) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: p.x, y: 1.45, w: 2.95, h: 5.0,
      fill: { color: T.white },
      shadow: { type: "outer", color: T.ink, blur: 10, offset: 3, opacity: 0.12, angle: 90 },
      rectRadius: 0.1,
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: p.x, y: 1.45, w: 2.95, h: 0.1,
      fill: { color: p.accent },
    });
    slide.addText(p.label, {
      x: p.x + 0.15, y: 1.65, w: 2.65, h: 0.4,
      fontSize: 16, bold: true, color: T.ink, fontFace: "Calibri",
      margin: 0, isTextBox: true, align: "center",
    });
  }

  // Arrows between panels
  for (const x of [3.32, 6.47, 9.62]) {
    slide.addShape(pres.shapes.RIGHT_ARROW, {
      x, y: 3.7, w: 0.26, h: 0.32,
      fill: { color: T.slate },
    });
  }

  // ---- Panel 1: two opposing regimes as icons ----
  // Two circles (opponents) facing a team diamond cluster
  slide.addShape(pres.shapes.OVAL, {
    x: 0.85, y: 2.35, w: 0.85, h: 0.85,
    fill: { color: T.coral },
  });
  slide.addText("A", {
    x: 0.85, y: 2.35, w: 0.85, h: 0.85,
    fontSize: 28, bold: true, color: T.white, align: "center", valign: "middle",
    fontFace: "Calibri", margin: 0, isTextBox: true,
  });
  slide.addShape(pres.shapes.OVAL, {
    x: 2.05, y: 2.35, w: 0.85, h: 0.85,
    fill: { color: T.gold },
  });
  slide.addText("B", {
    x: 2.05, y: 2.35, w: 0.85, h: 0.85,
    fontSize: 28, bold: true, color: T.white, align: "center", valign: "middle",
    fontFace: "Calibri", margin: 0, isTextBox: true,
  });
  // vs bar
  slide.addText("vs", {
    x: 0.7, y: 3.35, w: 2.35, h: 0.35,
    fontSize: 14, color: T.slate, align: "center", fontFace: "Calibri",
    margin: 0, isTextBox: true,
  });
  // blue team as three small dots
  for (const [dx, dy] of [[1.15, 3.85], [1.55, 3.85], [1.95, 3.85]]) {
    slide.addShape(pres.shapes.OVAL, {
      x: dx, y: dy, w: 0.35, h: 0.35,
      fill: { color: T.teal },
    });
  }
  slide.addText("one team, two worlds", {
    x: 0.55, y: 4.45, w: 2.65, h: 0.4,
    fontSize: 13, color: T.ink2, align: "center", fontFace: "Calibri",
    margin: 0, isTextBox: true,
  });
  slide.addText("strategy choice\nmust matter", {
    x: 0.55, y: 5.15, w: 2.65, h: 0.7,
    fontSize: 14, bold: true, color: T.ink, align: "center", fontFace: "Calibri",
    margin: 0, isTextBox: true,
  });

  // ---- Panel 2: dual symmetric branches ----
  // Two stacked "branch" blocks with attack/defend split
  const drawBranch = (y, fill, tag) => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 3.8, y, w: 2.45, h: 1.15,
      fill: { color: fill }, rectRadius: 0.08,
    });
    slide.addText(tag, {
      x: 3.8, y: y + 0.08, w: 2.45, h: 0.35,
      fontSize: 14, bold: true, color: T.white, align: "center",
      fontFace: "Calibri", margin: 0, isTextBox: true,
    });
    // mini A/D split
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 4.0, y: y + 0.5, w: 0.95, h: 0.45,
      fill: { color: T.white }, rectRadius: 0.05,
    });
    slide.addText("ATK", {
      x: 4.0, y: y + 0.5, w: 0.95, h: 0.45,
      fontSize: 12, bold: true, color: fill, align: "center", valign: "middle",
      fontFace: "Calibri", margin: 0, isTextBox: true,
    });
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 5.1, y: y + 0.5, w: 0.95, h: 0.45,
      fill: { color: T.white }, rectRadius: 0.05,
    });
    slide.addText("DEF", {
      x: 5.1, y: y + 0.5, w: 0.95, h: 0.45,
      fontSize: 12, bold: true, color: fill, align: "center", valign: "middle",
      fontFace: "Calibri", margin: 0, isTextBox: true,
    });
  };
  drawBranch(2.3, T.teal, "π_A");
  drawBranch(3.7, T.sea, "π_B");
  slide.addText("same roles both sides\nk = ceil(N/3)", {
    x: 3.7, y: 5.15, w: 2.65, h: 0.7,
    fontSize: 13, bold: true, color: T.ink, align: "center", fontFace: "Calibri",
    margin: 0, isTextBox: true,
  });

  // ---- Panel 3: funnel into shared student ----
  // Two small teacher nodes → funnel → one big shared node
  slide.addShape(pres.shapes.OVAL, {
    x: 7.0, y: 2.3, w: 0.55, h: 0.55,
    fill: { color: T.teal },
  });
  slide.addShape(pres.shapes.OVAL, {
    x: 8.8, y: 2.3, w: 0.55, h: 0.55,
    fill: { color: T.sea },
  });
  // chevron / funnel lines as triangles via right arrows stacked
  slide.addShape(pres.shapes.DOWN_ARROW, {
    x: 7.85, y: 2.95, w: 0.55, h: 0.55,
    fill: { color: T.slate },
  });
  // shared student
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.0, y: 3.65, w: 2.35, h: 1.2,
    fill: { color: T.ink2 }, rectRadius: 0.1,
  });
  slide.addText("π(a | o, z, r)", {
    x: 7.0, y: 3.75, w: 2.35, h: 0.55,
    fontSize: 16, bold: true, color: T.white, align: "center", valign: "middle",
    fontFace: "Calibri", margin: 0, isTextBox: true,
  });
  slide.addText("shared student", {
    x: 7.0, y: 4.3, w: 2.35, h: 0.4,
    fontSize: 12, color: T.ice, align: "center",
    fontFace: "Calibri", margin: 0, isTextBox: true,
  });
  // param callout
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.15, y: 5.15, w: 2.05, h: 0.7,
    fill: { color: T.slate }, rectRadius: 0.08,
  });
  slide.addText("~75% fewer\nactor params", {
    x: 7.15, y: 5.15, w: 2.05, h: 0.7,
    fontSize: 13, bold: true, color: T.white, align: "center", valign: "middle",
    fontFace: "Calibri", margin: 0, isTextBox: true,
  });

  // ---- Panel 4: crossover matrix visual ----
  // 2x2 win cells
  const cell = (x, y, fill, label, sub) => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 1.15, h: 1.15,
      fill: { color: fill }, rectRadius: 0.08,
    });
    slide.addText(label, {
      x, y: y + 0.2, w: 1.15, h: 0.4,
      fontSize: 14, bold: true, color: T.white, align: "center",
      fontFace: "Calibri", margin: 0, isTextBox: true,
    });
    slide.addText(sub, {
      x, y: y + 0.55, w: 1.15, h: 0.35,
      fontSize: 11, color: T.white, align: "center",
      fontFace: "Calibri", margin: 0, isTextBox: true,
    });
  };
  // intended diagonals highlighted
  cell(10.15, 2.35, T.teal, "A→A", "want high");
  cell(11.45, 2.35, T.slate, "B→A", "want low");
  cell(10.15, 3.65, T.slate, "A→B", "want low");
  cell(11.45, 3.65, T.sea, "B→B", "want high");
  slide.addText("Δ_A > 0  and  Δ_B > 0", {
    x: 10.0, y: 5.05, w: 2.65, h: 0.35,
    fontSize: 14, bold: true, color: T.ink, align: "center",
    fontFace: "Calibri", margin: 0, isTextBox: true,
  });
  slide.addText("both poles", {
    x: 10.0, y: 5.4, w: 2.65, h: 0.3,
    fontSize: 13, color: T.ink2, align: "center",
    fontFace: "Calibri", margin: 0, isTextBox: true,
  });

  // Footer — one line result
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.4, y: 6.65, w: 12.5, h: 0.6,
    fill: { color: T.ink }, rectRadius: 0.06,
  });
  slide.addText(
    "2v2: both sides  ·  4v4: A-side  ·  6v6: B-side  ·  ~75% fewer actor params",
    {
      x: 0.5, y: 6.65, w: 12.3, h: 0.6,
      fontSize: 15, bold: true, color: T.white, align: "center", valign: "middle",
      fontFace: "Calibri", margin: 0, isTextBox: true,
    }
  );

  const out = path.join(__dirname, "fig_visual_abstract.pptx");
  await pres.writeFile({ fileName: out });
  console.log("wrote", out);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
