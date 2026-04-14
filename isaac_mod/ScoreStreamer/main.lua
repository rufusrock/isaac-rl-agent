-- Binding of Isaac: Afterbirth+  --  ScoreStreamer (extended, ASCII-only)

local mod = RegisterMod("ScoreStreamerABPlusEXT", 1)

local udp = require("socket").udp()
udp:settimeout(0)
udp:setpeername("127.0.0.1", 8123)

----------------------------------------------------------------------
-- 1. State counters
----------------------------------------------------------------------
local roomsCleared, floorsCleared = 0, 0
local kills, dmgTaken            = 0, 0
local roomsExplored              = 0
local deaths                     = 0
local playerDead                 = false

-- Floor room graph: gridIndex -> {t=type, v=visitedCount, c=cleared(0/1)}
local floorRooms = {}

local function resetCounters()
  roomsCleared, floorsCleared = 0, 0
  kills, dmgTaken             = 0, 0
  roomsExplored, deaths       = 0, 0
  playerDead                  = false
end

local function buildFloorMap()
  floorRooms = {}
  local level = Game():GetLevel()
  -- Standard Isaac floor grid is 13 columns wide, rows 0-12 (indices 0-168)
  for i = 0, 168 do
    local desc = level:GetRoomByIdx(i)
    if desc and desc.Data and desc.Data.Type and desc.Data.Type > 0 then
      floorRooms[i] = {
        t = desc.Data.Type,
        v = desc.VisitedCount or 0,
        c = 0,
      }
    end
  end
end

----------------------------------------------------------------------
-- 2. Event hooks
----------------------------------------------------------------------
mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function()
  resetCounters()
  buildFloorMap()
end)

mod:AddCallback(ModCallbacks.MC_POST_NEW_LEVEL, function()
  floorsCleared = floorsCleared + 1
  floorRooms    = {}
  buildFloorMap()
end)

-- Update visited/cleared state whenever we enter a room
mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, function()
  local level   = Game():GetLevel()
  local idx     = level:GetCurrentRoomIndex()
  local desc    = level:GetCurrentRoomDesc()
  if not desc then return end

  -- Rooms with no enemies (shops, treasure, starting room, etc.)
  -- are immediately considered cleared for navigation purposes.
  local alreadyClear = Game():GetRoom():IsClear() and 1 or 0

  if floorRooms[idx] then
    floorRooms[idx].v = desc.VisitedCount or 0
    if alreadyClear == 1 then
      floorRooms[idx].c = 1
    end
  else
    -- Defensive: room not in map yet (e.g. first room on continue)
    if desc.Data and desc.Data.Type and desc.Data.Type > 0 then
      floorRooms[idx] = {t = desc.Data.Type, v = desc.VisitedCount or 0, c = alreadyClear}
    end
  end

  if desc.VisitedCount == 1 then
    roomsExplored = roomsExplored + 1
  end
end)

mod:AddCallback(ModCallbacks.MC_PRE_SPAWN_CLEAN_AWARD, function()
  roomsCleared = roomsCleared + 1
  local idx    = Game():GetLevel():GetCurrentRoomIndex()
  if floorRooms[idx] then
    floorRooms[idx].c = 1
  end
end)

mod:AddCallback(ModCallbacks.MC_POST_NPC_DEATH, function(_, npc)
  if npc:IsEnemy() and not npc:HasEntityFlags(EntityFlag.FLAG_FRIENDLY) then
    kills = kills + 1
  end
end)

mod:AddCallback(ModCallbacks.MC_POST_PLAYER_UPDATE, function(_, player)
  if player:IsDead() and player:GetExtraLives() == 0 then
    if not playerDead then
      deaths    = deaths + 1
      playerDead = true
    end
  else
    playerDead = false
  end
end)

mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, function(_, ent, amount)
  if ent:ToPlayer() then
    dmgTaken = dmgTaken + amount
  end
end, EntityType.ENTITY_PLAYER)

----------------------------------------------------------------------
-- 3. Helper: count passive items
----------------------------------------------------------------------
local function countCollectibles(pl)
  local n = 0
  for id = 1, CollectibleType.NUM_COLLECTIBLES - 1 do
    local c = pl:GetCollectibleNum(id)
    if c and c > 0 then n = n + c end
  end
  return n
end

----------------------------------------------------------------------
-- 4. Serialize room graph
----------------------------------------------------------------------
local function serializeRooms()
  local parts = {}
  for idx, info in pairs(floorRooms) do
    parts[#parts + 1] = idx .. ":" .. info.t .. ":" .. info.v .. ":" .. info.c
  end
  return table.concat(parts, ",")
end

----------------------------------------------------------------------
-- 5. Streamer  (every 4 game frames ~ 67 ms at 60 fps)
----------------------------------------------------------------------
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
  local g = Game()
  local f = g:GetFrameCount()
  if f % 4 ~= 0 then return end

  local p = Isaac.GetPlayer(0)
  if not p then return end

  -- Keep current room's cleared flag in sync every tick.
  -- IsClear() is unreliable in MC_POST_NEW_ROOM (room not fully loaded),
  -- so we update it here instead.
  local currentIdx = g:GetLevel():GetCurrentRoomIndex()
  if floorRooms[currentIdx] and g:GetRoom():IsClear() then
    floorRooms[currentIdx].c = 1
  end

  local scalars = table.concat({
    f,
    roomsCleared, floorsCleared,
    kills, dmgTaken,
    p:GetNumCoins(), p:GetNumKeys(), p:GetNumBombs(),
    p:GetHearts(), p:GetSoulHearts(), p:GetBlackHearts(),
    countCollectibles(p),
    g:GetLevel():GetCurrentRoomIndex(),
    g:GetLevel():GetStage(), roomsExplored, deaths,
  }, ",")

  udp:send(scalars .. "|" .. serializeRooms())
end)
