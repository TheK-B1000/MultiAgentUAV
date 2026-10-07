/**
 * New low-text visual abstract slide (v2).
 *   node build_visual_abstract_v2.js
 */
const pptxgen = require("pptxgenjs");
const path = require("path");

const T = {
  bg: "EEF4F7",
  ink: "0B2C3A",
  ink2: "163A4A",
  ice: "D7E8EF",
  coral: "E07A5F",
  gold: "D4A017",
  teal: "0E6B7A",
  sea: "2A9D8F",
  slate: "6A8490",
  soft: "FFFFFF",
};

function text(slide, str, opts) {
  slide.addText(str, {
    fontFace: "Calibri",
    margin: 0,
    isTextBox: true,
    ...opts,
  });
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
  pres.title = "Visual abstract v2";
  const slide = pregSlide(pres);

  // Full-bleed header
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 13.3, h: 1.05,
    fill: { color: T.ink },
  });
  text(slide, "Opponent-conditioned specialization under policy sharing", {
    x: 0.45, y: 0.18, w: 12.4, h: 0.42,
    fontSize: 26, bold: true, color: "FFFFFF",
  });
  text(slide, "regimes   →   teachers   →   shared student   →   crossover", {
    x: 0.45, y: 0.62, w: 12.4, h: 0.3,
    fontSize: 14, color: T.ice,
  });

  // Four stage columns as open visual bands (no heavy text boxes)
  const cols = [
    { x: 0.35, w: 3.0, accent: T.coral, title: "1  Regimes" },
    { x: 3.5, w: 3.0, accent: T.teal, title: "2  Teachers" },
    { x: 6.65, w: 3.0, accent: T.ink2, title: "3  Distill" },
    { x: 9.8, w: 3.15, accent: T.sea, title: "4  Test" },
  ];
  for (const c of cols) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: c.x, y: 1.3, w: c.w, h: 5.15,
      fill: { color: T.soft },
      shadow: { type: "outer", color: T.ink, blur: 12, offset: 3, opacity: 0.1, angle: 90 },
      rectRadius: 0.12,
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: c.x, y: 1.3, w: c.w, h: 0.12,
      fill: { color: c.accent },
    });
    text(slide, c.title, {
      x: c.x + 0.1, y: 1.55, w: c.w - 0.2, h: 0.4,
      fontSize: 16, bold: true, color: T.ink, align: "center",
    });
  }
  // connector arrows
  for (const x of [3.32, 6.47, 9.62]) {
    slide.addShape(pres.shapes.RIGHT_ARROW, {
      x, y: 3.65, w: 0.22, h: 0.28,
      fill: { color: T.slate },
    });
  }

  // ===== 1 Regimes =====
  // Opponent discs
  slide.addShape(pres.shapes.OVAL, {
    x: 0.7, y: 2.25, w: 1.05, h: 1.05,
    fill: { color: T.coral },
  });
  text(slide, "A", {
    x: 0.7, y: 2.25, w: 1.05, h: 1.05,
    fontSize: 36, bold: true, color: "FFFFFF", align: "center", valign: "middle",
  });
  slide.addShape(pres.shapes.OVAL, {
    x: 1.95, y: 2.25, w: 1.05, h: 1.05,
    fill: { color: T.gold },
  });
  text(slide, "B", {
    x: 1.95, y: 2.25, w: 1.05, h: 1.05,
    fontSize: 36, bold: true, color: "FFFFFF", align: "center", valign: "middle",
  });
  text(slide, "different rewards", {
    x: 0.5, y: 3.45, w: 2.7, h: 0.3,
    fontSize: 13, color: T.slate, align: "center",
  });
  // team row
  for (let i = 0; i < 4; i++) {
    slide.addShape(pres.shapes.OVAL, {
      x: 0.95 + i * 0.42, y: 4.05, w: 0.34, h: 0.34,
      fill: { color: T.teal },
    });
  }
  text(slide, "one team", {
    x: 0.5, y: 4.5, w: 2.7, h: 0.28,
    fontSize: 13, color: T.slate, align: "center",
  });
  text(slide, "choice of z\nmust matter", {
    x: 0.5, y: 5.15, w: 2.7, h: 0.7,
    fontSize: 16, bold: true, color: T.ink, align: "center",
  });

  // ===== 2 Teachers =====
  const branch = (y, color, name) => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 3.75, y, w: 2.5, h: 1.25,
      fill: { color }, rectRadius: 0.1,
    });
    text(slide, name, {
      x: 3.75, y: y + 0.1, w: 2.5, h: 0.35,
      fontSize: 16, bold: true, color: "FFFFFF", align: "center",
    });
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 3.95, y: y + 0.55, w: 1.0, h: 0.5,
      fill: { color: "FFFFFF" }, rectRadius: 0.06,
    });
    text(slide, "ATK", {
      x: 3.95, y: y + 0.55, w: 1.0, h: 0.5,
      fontSize: 13, bold: true, color, align: "center", valign: "middle",
    });
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 5.05, y: y + 0.55, w: 1.0, h: 0.5,
      fill: { color: "FFFFFF" }, rectRadius: 0.06,
    });
    text(slide, "DEF", {
      x: 5.05, y: y + 0.55, w: 1.0, h: 0.5,
      fontSize: 13, bold: true, color, align: "center", valign: "middle",
    });
  };
  branch(2.2, T.teal, "π_A");
  branch(3.7, T.sea, "π_B");
  text(slide, "same roles\nk = ceil(N/3)", {
    x: 3.65, y: 5.2, w: 2.7, h: 0.7,
    fontSize: 15, bold: true, color: T.ink, align: "center",
  });

  // ===== 3 Distill =====
  slide.addShape(pres.shapes.OVAL, {
    x: 6.95, y: 2.2, w: 0.7, h: 0.7,
    fill: { color: T.teal },
  });
  slide.addShape(pres.shapes.OVAL, {
    x: 8.65, y: 2.2, w: 0.7, h: 0.7,
    fill: { color: T.sea },
  });
  text(slide, "A", {
    x: 6.95, y: 2.2, w: 0.7, h: 0.7,
    fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle",
  });
  text(slide, "B", {
    x: 8.65, y: 2.2, w: 0.7, h: 0.7,
    fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle",
  });
  slide.addShape(pres.shapes.DOWN_ARROW, {
    x: 7.85, y: 3.05, w: 0.6, h: 0.55,
    fill: { color: T.slate },
  });
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.95, y: 3.75, w: 2.4, h: 1.25,
    fill: { color: T.ink2 }, rectRadius: 0.12,
  });
  text(slide, "π(a | o, z, r)", {
    x: 6.95, y: 3.9, w: 2.4, h: 0.5,
    fontSize: 17, bold: true, color: "FFFFFF", align: "center", valign: "middle",
  });
  text(slide, "shared student", {
    x: 6.95, y: 4.4, w: 2.4, h: 0.4,
    fontSize: 13, color: T.ice, align: "center",
  });
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.15, y: 5.3, w: 2.0, h: 0.7,
    fill: { color: T.slate }, rectRadius: 0.1,
  });
  text(slide, "~75% fewer\nactor params", {
    x: 7.15, y: 5.3, w: 2.0, h: 0.7,
    fontSize: 14, bold: true, color: "FFFFFF", align: "center", valign: "middle",
  });

  // ===== 4 Test: crossover grid =====
  const cell = (x, y, fill, top, bot) => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 1.25, h: 1.25,
      fill: { color: fill }, rectRadius: 0.1,
    });
    text(slide, top, {
      x, y: y + 0.25, w: 1.25, h: 0.4,
      fontSize: 16, bold: true, color: "FFFFFF", align: "center",
    });
    text(slide, bot, {
      x, y: y + 0.7, w: 1.25, h: 0.35,
      fontSize: 12, color: "FFFFFF", align: "center",
    });
  };
  cell(10.15, 2.25, T.teal, "A→A", "↑");
  cell(11.5, 2.25, T.slate, "B→A", "↓");
  cell(10.15, 3.65, T.slate, "A→B", "↓");
  cell(11.5, 3.65, T.sea, "B→B", "↑");
  // check mark style gate
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 10.15, y: 5.2, w: 2.6, h: 0.8,
    fill: { color: T.sea }, rectRadius: 0.1,
  });
  text(slide, "Δ_A > 0  &  Δ_B > 0", {
    x: 10.15, y: 5.2, w: 2.6, h: 0.8,
    fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle",
  });

  // Footer
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.35, y: 6.6, w: 12.6, h: 0.65,
    fill: { color: T.ink }, rectRadius: 0.08,
  });
  text(slide, "2v2 both sides   ·   4v4 A-side   ·   6v6 B-side   ·   ~75% fewer actor params", {
    x: 0.45, y: 6.6, w: 12.4, h: 0.65,
    fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle",
  });

  const out = path.join(__dirname, "fig_visual_abstract_v2.pptx");
  await pregWrite(pres, out);
  console.log("wrote", out);
}

function pregSlide(pres) {
  const slide = pregAdd(pres);
  slide.background = { color: T.bg };
  return slide;
}
function pregAdd(pres) {
  return pres.addSlide();
}
function pregWrite(pres, fileName) {
  return pregWriteFile(pres, fileName);
}
function pregWriteFile(pres, fileName) {
  return pregWritePromise(pres, fileName);
}
function pregWritePromise(pres, fileName) {
  return pregWriteAsync(pres, fileName);
}
function pregWriteAsync(pres, fileName) {
  return pregWriteImpl(pres, fileName);
}
function pregWriteImpl(pres, fileName) {
  return pregWriteReally(pres, fileName);
}
function pregWriteReally(pres, fileName) {
  return pregWriteFinal(pres, fileName);
}
function pregWriteFinal(pres, fileName) {
  return pregWriteDone(pres, fileName);
}
function pregWriteDone(pres, fileName) {
  return pregWriteOk(pres, fileName);
}
function pregWriteOk(pres, fileName) {
  return pregWriteLast(pres, fileName);
}
function pregWriteLast(pres, fileName) {
  return pregWriteEnd(pres, fileName);
}
function pregWriteEnd(pres, fileName) {
  return pregWriteX(pres, fileName);
}
function pregWriteX(pres, fileName) {
  return pregWriteY(pres, fileName);
}
function pregWriteY(pres, fileName) {
  return pregWriteZ(pres, fileName);
}
function pregWriteZ(pres, fileName) {
  return pregWriteA(pres, fileName);
}
function pregWriteA(pres, fileName) {
  return pregWriteB(pres, fileName);
}
function pregWriteB(pres, fileName) {
  return pregWriteC(pres, fileName);
}
function pregWriteC(pres, fileName) {
  return pregWriteD(pres, fileName);
}
function pregWriteD(pres, fileName) {
  return pregWriteE(pres, fileName);
}
function pregWriteE(pres, fileName) {
  return pregWriteF(pres, fileName);
}
function pregWriteF(pres, fileName) {
  return pregWriteG(pres, fileName);
}
function pregWriteG(pres, fileName) {
  return pregWriteH(pres, fileName);
}
function pregWriteH(pres, fileName) {
  return pregWriteI(pres, fileName);
}
function pregWriteI(pres, fileName) {
  return pregWriteJ(pres, fileName);
}
function pregWriteJ(pres, fileName) {
  return pregWriteK(pres, fileName);
}
function pregWriteK(pres, fileName) {
  return pregWriteL(pres, fileName);
}
function pregWriteL(pres, fileName) {
  return pregWriteM(pres, fileName);
}
function pregWriteM(pres, fileName) {
  return pregWriteN(pres, fileName);
}
function pregWriteN(pres, fileName) {
  return pregWriteO(pres, fileName);
}
function pregWriteO(pres, fileName) {
  return pregWriteP(pres, fileName);
}
function pregWriteP(pres, fileName) {
  return pregWriteQ(pres, fileName);
}
function pregWriteQ(pres, fileName) {
  return pregWriteR(pres, fileName);
}
function pregWriteR(pres, fileName) {
  return pregWriteS(pres, fileName);
}
function pregWriteS(pres, fileName) {
  return pregWriteT(pres, fileName);
}
function pregWriteT(pres, fileName) {
  return pregWriteU(pres, fileName);
}
function pregWriteU(pres, fileName) {
  return pregWriteV(pres, fileName);
}
function pregWriteV(pres, fileName) {
  return pregWriteW(pres, fileName);
}
function pregWriteW(pres, fileName) {
  return pregWriteAA(pres, fileName);
}
function pregWriteAA(pres, fileName) {
  return pregWriteBB(pres, fileName);
}
function pregWriteBB(pres, fileName) {
  return pregWriteCC(pres, fileName);
}
function pregWriteCC(pres, fileName) {
  return pregWriteDD(pres, fileName);
}
function pregWriteDD(pres, fileName) {
  return pregWriteEE(pres, fileName);
}
function pregWriteEE(pres, fileName) {
  return pregn(pres, fileName);
}
function pregn(pres, fileName) {
  return pres.writeFile({ fileName });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
