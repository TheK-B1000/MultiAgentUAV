/**
 * Scenario + objectives figure with maritime imagery.
 *   node build_scenario_feel.js
 * Writes: fig_scenario_feel.pptx / .png (png via PowerPoint export if available;
 *         otherwise pptx only — open and export, or use LibreOffice).
 */
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const DIR = __dirname;
const T = {
  bg: "F2F7F9",
  ink: "0B2C3A",
  ink2: "163A4A",
  ice: "D7E8EF",
  coral: "E07A5F",
  gold: "D4A017",
  teal: "0E6B7A",
  sea: "2A9D8F",
  slate: "5C7A86",
  white: "FFFFFF",
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
  const imgOverview = path.join(DIR, "scenario_maritime_ctf_overview.jpg");
  const imgRegimes = path.join(DIR, "scenario_regime_ab_boats.jpg");
  const imgRoles = path.join(DIR, "scenario_roles_atk_def.jpg");
  for (const p of [imgOverview, imgRegimes, imgRoles]) {
    if (!fs.existsSync(p)) throw new Error("missing " + p);
  }

  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
  pres.title = "Maritime CTF scenario and objectives";
  const slide = pres.addSlide();
  slide.background = { color: T.bg };

  // Header
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 13.3, h: 0.95,
    fill: { color: T.ink },
  });
  text(slide, "Maritime Capture-the-Flag: scenario and objectives", {
    x: 0.4, y: 0.16, w: 12.5, h: 0.4,
    fontSize: 24, bold: true, color: T.white,
  });
  text(slide, "One Blue team of ASVs  ·  two Red opponent regimes  ·  specialize strategy without duplicating the whole policy", {
    x: 0.4, y: 0.55, w: 12.5, h: 0.28,
    fontSize: 13, color: T.ice,
  });

  // Left: overview photo
  slide.addImage({
    path: imgOverview,
    x: 0.3, y: 1.15, w: 6.3, h: 3.55,
  });
  text(slide, "Task: Blue ASVs must capture Red's flag and return home under partial observability.", {
    x: 0.35, y: 4.75, w: 6.2, h: 0.35,
    fontSize: 12, color: T.ink2,
  });

  // Right top: regimes
  slide.addImage({
    path: imgRegimes,
    x: 6.8, y: 1.15, w: 6.2, h: 2.55,
  });

  // Right bottom: roles
  slide.addImage({
    path: imgRoles,
    x: 6.8, y: 3.85, w: 6.2, h: 2.35,
  });

  // Bottom objective strip
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.3, y: 5.2, w: 6.3, h: 2.0,
    fill: { color: T.white },
    shadow: { type: "outer", color: T.ink, blur: 8, offset: 2, opacity: 0.1, angle: 90 },
    rectRadius: 0.08,
  });
  text(slide, "Objectives", {
    x: 0.5, y: 5.35, w: 5.9, h: 0.32,
    fontSize: 15, bold: true, color: T.teal,
  });
  text(slide, [
    { text: "Regime A ", options: { bold: true, color: T.coral } },
    { text: "(broad / passive zone defense) favors a GUARD-style Blue response.\n", options: { color: T.ink } },
    { text: "Regime B ", options: { bold: true, color: T.gold } },
    { text: "(compact / reactive defense) favors a BREACH-style Blue response.\n", options: { color: T.ink } },
    { text: "Learn complementary strategies, then compress them into one shared student ", options: { color: T.ink } },
    { text: "π(a | o, z, r)", options: { bold: true, color: T.ink2 } },
    { text: " so the correct strategy wins on each pole.", options: { color: T.ink } },
  ], {
    x: 0.5, y: 5.7, w: 5.9, h: 1.3,
    fontSize: 12, valign: "top",
  });

  // Tiny legend under roles image area is already in image; add footer
  text(slide, "Assets: scenario_maritime_ctf_overview.jpg · scenario_regime_ab_boats.jpg · scenario_roles_atk_def.jpg", {
    x: 0.3, y: 7.2, w: 12.7, h: 0.22,
    fontSize: 9, color: T.slate,
  });

  const out = path.join(DIR, "fig_scenario_feel.pptx");
  await pres.writeFile({ fileName: out });
  console.log("wrote", out);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
